from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch


class GaussianDataset:
    def __init__(
        self,
        sparse_path,
        image_path,
        config,
        device="cuda",
    ):
        self.sparse_path = Path(sparse_path)
        self.image_path = Path(image_path)

        self.config = config
        self.device = torch.device(device)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Gaussian training."
            )

        self.max_resolution = int(
            config["gaussian"]["training"]["max_resolution"]
        )

        if not self.sparse_path.exists():
            raise FileNotFoundError(
                f"Sparse reconstruction not found: {self.sparse_path}"
            )

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_path}"
            )

        self.reconstruction = pycolmap.Reconstruction(
            str(self.sparse_path)
        )

        # Cache of per-camera undistortion maps, keyed by
        # COLMAP camera_id. See _get_undistort_map().
        self._undistort_maps = {}

        self.frames = self._build_frames()

    # ---------------------------------------------------------
    # UNDISTORTION
    # ---------------------------------------------------------
    #
    # COLMAP's SIMPLE_RADIAL/RADIAL camera models carry real
    # lens distortion coefficients (k1, k2). gsplat's
    # rasterizer only supports an ideal, distortion-free
    # "pinhole" camera model. If those coefficients are
    # dropped (as the old _camera_parameters() did), every
    # training step compares an ideal synthetic render against
    # a photo that still has real lens distortion in it -- a
    # mismatch that grows toward the frame edges/corners and
    # shows up as persistent softness/ghosting there no matter
    # how long training runs.
    #
    # Fix: undistort each image once with OpenCV before it
    # enters training, and report the corresponding corrected
    # (already-pinhole) intrinsics to the renderer instead of
    # the original distorted ones. Maps are computed once per
    # camera and cached, since most captures share one camera
    # for every frame (camera.shared=true).
    # ---------------------------------------------------------

    def _get_undistort_map(
        self,
        camera_id,
        camera,
        fx,
        fy,
        cx,
        cy,
        dist_coeffs,
    ):
        if camera_id in self._undistort_maps:
            return self._undistort_maps[camera_id]

        width = int(self._value(camera, "width"))
        height = int(self._value(camera, "height"))

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # OpenCV expects [k1, k2, p1, p2, k3]. COLMAP's
        # RADIAL/SIMPLE_RADIAL models have no tangential (p1,
        # p2) terms, so those are zero.
        cv_dist = np.zeros(5, dtype=np.float64)
        cv_dist[: len(dist_coeffs)] = dist_coeffs

        if np.allclose(cv_dist, 0.0):
            # No real distortion (PINHOLE/SIMPLE_PINHOLE, or a
            # RADIAL camera COLMAP fit with ~zero k1/k2) -- no
            # remapping needed, identity intrinsics.
            self._undistort_maps[camera_id] = None
            return None

        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K,
            cv_dist,
            (width, height),
            alpha=0.0,
            newImgSize=(width, height),
        )

        map_x, map_y = cv2.initUndistortRectifyMap(
            K,
            cv_dist,
            None,
            new_K,
            (width, height),
            cv2.CV_32FC1,
        )

        result = {
            "map_x": map_x,
            "map_y": map_y,
            "fx": float(new_K[0, 0]),
            "fy": float(new_K[1, 1]),
            "cx": float(new_K[0, 2]),
            "cy": float(new_K[1, 2]),
        }

        self._undistort_maps[camera_id] = result

        return result

    # ---------------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------------

    def _load_image(self, path, undistort_map):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        if undistort_map is not None:
            image = cv2.remap(
                image,
                undistort_map["map_x"],
                undistort_map["map_y"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = image.shape[:2]

        scale = min(
            1.0,
            self.max_resolution
            / max(original_width, original_height),
        )

        if scale < 1.0:
            new_width = max(
                1,
                int(round(original_width * scale)),
            )

            new_height = max(
                1,
                int(round(original_height * scale)),
            )

            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        tensor = torch.from_numpy(
            image.copy()
        ).float()

        tensor /= 255.0

        return tensor.to(self.device)

    # ---------------------------------------------------------
    # SAFE ATTRIBUTE / METHOD HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _value(obj, name):
        """
        pycolmap versions differ:
        some fields are properties,
        some are callable methods.

        This helper supports both.
        """

        value = getattr(obj, name)

        if callable(value):
            return value()

        return value

    # ---------------------------------------------------------
    # CAMERA PARAMETERS
    # ---------------------------------------------------------

    def _camera_parameters(self, camera):
        params = np.asarray(
            camera.params,
            dtype=np.float32,
        )

        model = str(camera.model)

        # pycolmap may return:
        # CameraModelId.SIMPLE_RADIAL
        # or
        # SIMPLE_RADIAL

        # dist_coeffs is [k1] or [k1, k2] depending on model;
        # empty for models with no radial distortion term.
        # These are no longer discarded -- see
        # _get_undistort_map() for why that mattered.

        if model.endswith("SIMPLE_RADIAL"):
            if len(params) < 4:
                raise ValueError(
                    "SIMPLE_RADIAL camera has invalid parameters: "
                    f"{params}"
                )

            focal = float(params[0])
            cx = float(params[1])
            cy = float(params[2])

            fx = focal
            fy = focal

            dist_coeffs = [float(params[3])]

        elif model.endswith("PINHOLE"):
            if len(params) < 4:
                raise ValueError(
                    "PINHOLE camera has invalid parameters: "
                    f"{params}"
                )

            fx = float(params[0])
            fy = float(params[1])
            cx = float(params[2])
            cy = float(params[3])

            dist_coeffs = []

        elif model.endswith("SIMPLE_PINHOLE"):
            if len(params) < 3:
                raise ValueError(
                    "SIMPLE_PINHOLE camera has invalid parameters: "
                    f"{params}"
                )

            focal = float(params[0])
            cx = float(params[1])
            cy = float(params[2])

            fx = focal
            fy = focal

            dist_coeffs = []

        elif model.endswith("RADIAL"):
            if len(params) < 5:
                raise ValueError(
                    "RADIAL camera has invalid parameters: "
                    f"{params}"
                )

            focal = float(params[0])
            cx = float(params[1])
            cy = float(params[2])

            fx = focal
            fy = focal

            dist_coeffs = [
                float(params[3]),
                float(params[4]),
            ]

        else:
            raise ValueError(
                f"Unsupported camera model: {model}"
            )

        return (
            fx,
            fy,
            cx,
            cy,
            dist_coeffs,
        )

    # ---------------------------------------------------------
    # CAMERA LOOKUP
    # ---------------------------------------------------------

    def _get_camera(self, camera_id):
        """
        CameraMap does not implement .get() in this pycolmap
        version.

        Use [] lookup instead.
        """

        try:
            return self.reconstruction.cameras[camera_id]
        except Exception as exc:
            raise RuntimeError(
                f"Unable to find camera ID {camera_id} "
                f"in sparse reconstruction."
            ) from exc

    # ---------------------------------------------------------
    # CAMERA-FROM-WORLD
    # ---------------------------------------------------------

    def _get_cam_from_world(self, image):
        """
        Different pycolmap releases expose cam_from_world
        differently.

        It may be:
            image.cam_from_world

        or:
            image.cam_from_world()

        Handle both.
        """

        value = getattr(
            image,
            "cam_from_world",
        )

        if callable(value):
            value = value()

        return value

    # ---------------------------------------------------------
    # ROTATION
    # ---------------------------------------------------------

    def _get_rotation(self, cam_from_world):
        """
        Supports both:

            cam_from_world.rotation

        and:

            cam_from_world.rotation()

        """

        rotation = getattr(
            cam_from_world,
            "rotation",
        )

        if callable(rotation):
            rotation = rotation()

        matrix = getattr(
            rotation,
            "matrix",
        )

        if callable(matrix):
            matrix = matrix()

        return np.asarray(
            matrix,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # TRANSLATION
    # ---------------------------------------------------------

    def _get_translation(self, cam_from_world):
        """
        Supports both property and method versions.
        """

        translation = getattr(
            cam_from_world,
            "translation",
        )

        if callable(translation):
            translation = translation()

        return np.asarray(
            translation,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # BUILD TRAINING FRAMES
    # ---------------------------------------------------------

    def _build_frames(self):
        frames = []

        print(
            "Building Gaussian training frames..."
        )

        for image_id, image in self.reconstruction.images.items():

            # -------------------------------------------------
            # Check pose
            # -------------------------------------------------

            has_pose = getattr(
                image,
                "has_pose",
                False,
            )

            if callable(has_pose):
                has_pose = has_pose()

            if not has_pose:
                continue

            # -------------------------------------------------
            # Camera
            # -------------------------------------------------

            camera_id = getattr(
                image,
                "camera_id",
            )

            if callable(camera_id):
                camera_id = camera_id()

            camera = self._get_camera(
                camera_id
            )

            # -------------------------------------------------
            # Image name
            # -------------------------------------------------

            image_name = getattr(
                image,
                "name",
            )

            if callable(image_name):
                image_name = image_name()

            image_name = str(image_name)

            image_file = (
                self.image_path
                / image_name
            )

            if not image_file.exists():
                continue

            # -------------------------------------------------
            # Camera parameters (may include real distortion)
            # -------------------------------------------------

            (
                fx,
                fy,
                cx,
                cy,
                dist_coeffs,
            ) = self._camera_parameters(
                camera
            )

            # -------------------------------------------------
            # Original camera dimensions
            # -------------------------------------------------

            camera_width = getattr(
                camera,
                "width",
            )

            if callable(camera_width):
                camera_width = camera_width()

            camera_height = getattr(
                camera,
                "height",
            )

            if callable(camera_height):
                camera_height = camera_height()

            camera_width = int(camera_width)
            camera_height = int(camera_height)

            # -------------------------------------------------
            # Undistortion map (cached per camera_id) and
            # corrected pinhole intrinsics. If there is no
            # real distortion, this returns None and fx/fy/
            # cx/cy pass through unchanged.
            # -------------------------------------------------

            undistort_map = self._get_undistort_map(
                camera_id,
                camera,
                fx,
                fy,
                cx,
                cy,
                dist_coeffs,
            )

            if undistort_map is not None:
                fx = undistort_map["fx"]
                fy = undistort_map["fy"]
                cx = undistort_map["cx"]
                cy = undistort_map["cy"]

            # -------------------------------------------------
            # COLMAP camera pose
            # -------------------------------------------------

            cam_from_world = (
                self._get_cam_from_world(
                    image
                )
            )

            rotation = self._get_rotation(
                cam_from_world
            )

            translation = self._get_translation(
                cam_from_world
            )

            # -------------------------------------------------
            # Validate matrices
            # -------------------------------------------------

            if rotation.shape != (3, 3):
                raise RuntimeError(
                    f"Invalid rotation matrix for "
                    f"image {image_id}: "
                    f"shape={rotation.shape}"
                )

            if translation.shape != (3,):
                translation = translation.reshape(3)

            # -------------------------------------------------
            # Store frame
            # -------------------------------------------------

            frames.append(
                {
                    "image_id": int(image_id),

                    "name": image_name,

                    "image_path": image_file,

                    "width": camera_width,

                    "height": camera_height,

                    "fx": fx,

                    "fy": fy,

                    "cx": cx,

                    "cy": cy,

                    "rotation": rotation,

                    "translation": translation,

                    "undistort_map": undistort_map,
                }
            )

        # -----------------------------------------------------
        # Sort by COLMAP image ID
        # -----------------------------------------------------

        frames.sort(
            key=lambda frame: frame["image_id"]
        )

        if not frames:
            raise RuntimeError(
                "No posed images found in sparse reconstruction."
            )

        print(
            f"Gaussian training frames: {len(frames)}"
        )

        return frames

    # ---------------------------------------------------------
    # DATASET LENGTH
    # ---------------------------------------------------------

    def __len__(self):
        return len(self.frames)

    # ---------------------------------------------------------
    # GET TRAINING SAMPLE
    # ---------------------------------------------------------

    def __getitem__(self, index):
        frame = self.frames[index]

        image = self._load_image(
            frame["image_path"],
            frame["undistort_map"],
        )

        height, width = image.shape[:2]

        # -----------------------------------------------------
        # Image scaling
        # -----------------------------------------------------

        scale_x = (
            width
            / float(frame["width"])
        )

        scale_y = (
            height
            / float(frame["height"])
        )

        # -----------------------------------------------------
        # Scaled intrinsics
        # -----------------------------------------------------

        fx = frame["fx"] * scale_x
        fy = frame["fy"] * scale_y

        cx = frame["cx"] * scale_x
        cy = frame["cy"] * scale_y

        # -----------------------------------------------------
        # Pose tensors
        # -----------------------------------------------------

        rotation = torch.from_numpy(
            frame["rotation"].copy()
        ).float().to(
            self.device
        )

        translation = torch.from_numpy(
            frame["translation"].copy()
        ).float().to(
            self.device
        )

        # -----------------------------------------------------
        # Final training frame
        # -----------------------------------------------------

        return {
            "image": image,

            "image_id": frame[
                "image_id"
            ],

            "name": frame[
                "name"
            ],

            "width": int(width),

            "height": int(height),

            "fx": float(fx),

            "fy": float(fy),

            "cx": float(cx),

            "cy": float(cy),

            "rotation": rotation,

            "translation": translation,
        }