import inspect

import torch
from gsplat import rasterization


class GaussianRenderer:

    def __init__(
        self,
        config,
    ):

        self.config = config

        # =========================================================
        # DEVICE
        # =========================================================

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Gaussian rendering."
            )

        self.device = torch.device(
            "cuda"
        )

        # =========================================================
        # CONFIGURATION
        # =========================================================

        gaussian_config = config.get(
            "gaussian",
            {},
        )

        # =========================================================
        # RASTERIZATION SETTINGS
        # =========================================================

        self.radius_clip = float(
            gaussian_config.get(
                "radius_clip",
                0.0,
            )
        )

        self.eps2d = float(
            gaussian_config.get(
                "eps2d",
                0.3,
            )
        )

        self.near_plane = float(
            gaussian_config.get(
                "near_plane",
                0.01,
            )
        )

        self.far_plane = float(
            gaussian_config.get(
                "far_plane",
                1000.0,
            )
        )

        self.rasterize_mode = str(
            gaussian_config.get(
                "rasterize_mode",
                "classic",
            )
        ).lower()

        if self.rasterize_mode not in (
            "classic",
            "antialiased",
        ):
            raise ValueError(
                "gaussian.rasterize_mode must be "
                "'classic' or 'antialiased'."
            )

        # =========================================================
        # CAMERA MODEL
        # =========================================================
        #
        # gsplat 1.5.3 supports:
        #
        #   pinhole
        #   ortho
        #   fisheye
        #   ftheta
        #
        # Do NOT allow lidar here. It is not part of the
        # gsplat 1.5.3 rasterization API being used by this
        # project.
        #
        # =========================================================

        self.camera_model = str(
            gaussian_config.get(
                "camera_model",
                "pinhole",
            )
        ).lower()

        if self.camera_model not in (
            "pinhole",
            "ortho",
            "fisheye",
            "ftheta",
        ):
            raise ValueError(
                "Unsupported gaussian.camera_model: "
                f"{self.camera_model}. "
                "Supported models are: "
                "pinhole, ortho, fisheye, ftheta."
            )

        # =========================================================
        # PACKED MODE
        # =========================================================

        self.packed = bool(
            gaussian_config.get(
                "packed",
                True,
            )
        )

        # =========================================================
        # SPARSE GRADIENT
        # =========================================================

        self.sparse_grad = bool(
            gaussian_config.get(
                "sparse_grad",
                False,
            )
        )

        # =========================================================
        # TILE SIZE
        # =========================================================

        self.tile_size = int(
            gaussian_config.get(
                "tile_size",
                16,
            )
        )

        if self.tile_size <= 0:
            raise ValueError(
                "gaussian.tile_size must be "
                "greater than zero."
            )

        # =========================================================
        # CHANNEL CHUNK
        # =========================================================

        self.channel_chunk = int(
            gaussian_config.get(
                "channel_chunk",
                32,
            )
        )

        if self.channel_chunk <= 0:
            raise ValueError(
                "gaussian.channel_chunk must be "
                "greater than zero."
            )

        # =========================================================
        # BACKGROUND
        # =========================================================
        #
        # gsplat 1.5.3 has a known problem with:
        #
        #     packed=True
        #     backgrounds=[1, 3]
        #
        # This is exactly the assertion encountered in the
        # project:
        #
        #     AssertionError:
        #     torch.Size([1, 3])
        #
        # Therefore:
        #
        #   BLACK background -> do not pass backgrounds at all.
        #
        # This allows gsplat to use its native zero background
        # and keeps packed=True.
        #
        #   NON-BLACK background -> pass [1, 3] and force
        #   packed=False.
        #
        # This avoids the gsplat 1.5.3 packed/background issue.
        #
        # =========================================================

        background = gaussian_config.get(
            "background",
            [0.0, 0.0, 0.0],
        )

        if not isinstance(
            background,
            (list, tuple),
        ):
            raise ValueError(
                "gaussian.background must be "
                "a list or tuple."
            )

        if len(background) != 3:
            raise ValueError(
                "gaussian.background must contain "
                "exactly three values."
            )

        self.background = torch.tensor(
            [
                float(background[0]),
                float(background[1]),
                float(background[2]),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.background = torch.clamp(
            self.background,
            0.0,
            1.0,
        )

        self.has_custom_background = not bool(
            torch.allclose(
                self.background,
                torch.zeros(
                    3,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        )

        # =========================================================
        # PACKED/BACKGROUND COMPATIBILITY
        # =========================================================

        if self.has_custom_background:
            self.packed = False

        # =========================================================
        # ABSOLUTE GRADIENT
        # =========================================================

        self.absgrad = bool(
            gaussian_config.get(
                "absgrad",
                True,
            )
        )

        # =========================================================
        # OPTIONAL OUTPUT SETTINGS
        # =========================================================

        self.return_depth = bool(
            gaussian_config.get(
                "return_depth",
                False,
            )
        )

        self.return_normals = bool(
            gaussian_config.get(
                "return_normals",
                False,
            )
        )

        # =========================================================
        # NUMERICAL VALIDATION
        # =========================================================

        if self.radius_clip < 0:
            raise ValueError(
                "gaussian.radius_clip must be >= 0."
            )

        if self.eps2d <= 0:
            raise ValueError(
                "gaussian.eps2d must be > 0."
            )

        if self.near_plane <= 0:
            raise ValueError(
                "gaussian.near_plane must be > 0."
            )

        if self.far_plane <= self.near_plane:
            raise ValueError(
                "gaussian.far_plane must be greater "
                "than gaussian.near_plane."
            )

        # =========================================================
        # GSPLAT API DISCOVERY
        # =========================================================
        #
        # Keep this renderer explicitly compatible with the
        # installed gsplat API instead of blindly passing
        # unsupported arguments.
        #
        # =========================================================

        self._rasterization_signature = inspect.signature(
            rasterization
        )

        self._supported_arguments = set(
            self._rasterization_signature.parameters.keys()
        )

        required_arguments = {
            "means",
            "quats",
            "scales",
            "opacities",
            "colors",
            "viewmats",
            "Ks",
            "width",
            "height",
        }

        missing_arguments = (
            required_arguments
            - self._supported_arguments
        )

        if missing_arguments:
            raise RuntimeError(
                "Installed gsplat rasterization API is "
                "missing required arguments: "
                f"{sorted(missing_arguments)}"
            )

    # =============================================================
    # CAMERA INTRINSICS
    # =============================================================

    def _camera_matrix(
        self,
        frame,
    ):

        required = (
            "fx",
            "fy",
            "cx",
            "cy",
        )

        for key in required:

            if key not in frame:
                raise KeyError(
                    f"Frame is missing camera parameter "
                    f"'{key}'."
                )

        fx = float(
            frame["fx"]
        )

        fy = float(
            frame["fy"]
        )

        cx = float(
            frame["cx"]
        )

        cy = float(
            frame["cy"]
        )

        if not torch.isfinite(
            torch.tensor(
                [
                    fx,
                    fy,
                    cx,
                    cy,
                ],
                dtype=torch.float32,
            )
        ).all():

            raise ValueError(
                "Camera intrinsics contain "
                "non-finite values."
            )

        if fx <= 0:
            raise ValueError(
                f"Invalid focal length fx={fx}."
            )

        if fy <= 0:
            raise ValueError(
                f"Invalid focal length fy={fy}."
            )

        return torch.tensor(
            [
                [
                    fx,
                    0.0,
                    cx,
                ],
                [
                    0.0,
                    fy,
                    cy,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    # =============================================================
    # VIEW MATRIX
    # =============================================================

    def _view_matrix(
        self,
        frame,
    ):

        if "rotation" not in frame:
            raise KeyError(
                "Frame is missing camera rotation."
            )

        if "translation" not in frame:
            raise KeyError(
                "Frame is missing camera translation."
            )

        rotation = torch.as_tensor(
            frame["rotation"],
            dtype=torch.float32,
            device=self.device,
        )

        translation = torch.as_tensor(
            frame["translation"],
            dtype=torch.float32,
            device=self.device,
        )

        if rotation.shape != (
            3,
            3,
        ):
            raise ValueError(
                "Camera rotation must have shape "
                "[3, 3]. "
                f"Received {tuple(rotation.shape)}."
            )

        if translation.numel() != 3:
            raise ValueError(
                "Camera translation must contain "
                "exactly three values."
            )

        translation = translation.reshape(
            3
        )

        if not torch.isfinite(
            rotation
        ).all():

            raise ValueError(
                "Camera rotation contains "
                "non-finite values."
            )

        if not torch.isfinite(
            translation
        ).all():

            raise ValueError(
                "Camera translation contains "
                "non-finite values."
            )

        # =========================================================
        # COLMAP -> GSPLAT
        # =========================================================
        #
        # COLMAP stores:
        #
        #     X_camera = R * X_world + t
        #
        # gsplat expects a world-to-camera view matrix.
        #
        # Therefore:
        #
        #     [ R | t ]
        #     [ 0 | 1 ]
        #
        # is passed directly.
        #
        # Do NOT invert it here.
        #
        # =========================================================

        viewmat = torch.eye(
            4,
            dtype=torch.float32,
            device=self.device,
        )

        viewmat[
            :3,
            :3,
        ] = rotation

        viewmat[
            :3,
            3,
        ] = translation

        return viewmat

    # =============================================================
    # PARAMETER VALIDATION
    # =============================================================

    def _validate_parameters(
        self,
        parameters,
    ):

        required = (
            "means",
            "scales",
            "opacities",
        )

        for key in required:

            if key not in parameters:
                raise KeyError(
                    "Gaussian parameters are missing "
                    f"'{key}'."
                )

        # ---------------------------------------------------------
        # Rotation parameter
        #
        # Your current GaussianOptimizer creates:
        #
        #     parameters["rotations"]
        #
        # Older renderer versions expected:
        #
        #     parameters["quats"]
        #
        # Support both names.
        # ---------------------------------------------------------

        if (
            "rotations" not in parameters
            and "quats" not in parameters
        ):

            raise KeyError(
                "Gaussian parameters must contain "
                "'rotations' or 'quats'."
            )

        # ---------------------------------------------------------
        # Appearance
        #
        # Current project uses:
        #
        #     parameters["colors"]
        #
        # Optional SH support is retained for compatibility.
        # ---------------------------------------------------------

        if (
            "colors" not in parameters
            and "shs" not in parameters
        ):

            raise KeyError(
                "Gaussian parameters must contain "
                "'colors' or 'shs'."
            )

        means = parameters[
            "means"
        ]

        scales = parameters[
            "scales"
        ]

        opacities = parameters[
            "opacities"
        ]

        if means.ndim != 2:
            raise ValueError(
                "parameters['means'] must have "
                "shape [N, 3]."
            )

        if means.shape[-1] != 3:
            raise ValueError(
                "parameters['means'] must have "
                "shape [N, 3]."
            )

        if scales.ndim != 2:
            raise ValueError(
                "parameters['scales'] must have "
                "shape [N, 3]."
            )

        if scales.shape[-1] != 3:
            raise ValueError(
                "parameters['scales'] must have "
                "shape [N, 3]."
            )

        rotations = parameters.get(
            "rotations",
            parameters.get(
                "quats"
            ),
        )

        if rotations.ndim != 2:
            raise ValueError(
                "Gaussian rotations must have "
                "shape [N, 4]."
            )

        if rotations.shape[-1] != 4:
            raise ValueError(
                "Gaussian rotations must have "
                "shape [N, 4]."
            )

        if opacities.ndim not in (
            1,
            2,
        ):
            raise ValueError(
                "Gaussian opacities must have "
                "shape [N] or [N, 1]."
            )

        gaussian_count = means.shape[0]

        if gaussian_count == 0:
            raise RuntimeError(
                "Gaussian parameter set contains "
                "zero Gaussians."
            )

        if scales.shape[0] != gaussian_count:
            raise ValueError(
                "Gaussian means and scales have "
                "different numbers of Gaussians."
            )

        if rotations.shape[0] != gaussian_count:
            raise ValueError(
                "Gaussian means and rotations have "
                "different numbers of Gaussians."
            )

        if opacities.shape[0] != gaussian_count:
            raise ValueError(
                "Gaussian means and opacities have "
                "different numbers of Gaussians."
            )

        for name in (
            "means",
            "scales",
            "opacities",
        ):

            tensor = parameters[
                name
            ]

            if not torch.isfinite(
                tensor
            ).all():

                raise RuntimeError(
                    f"Gaussian parameter '{name}' "
                    "contains non-finite values."
                )

        if not torch.isfinite(
            rotations
        ).all():

            raise RuntimeError(
                "Gaussian rotations contain "
                "non-finite values."
            )

    # =============================================================
    # SCALES
    # =============================================================

    def _prepare_scales(
        self,
        parameters,
    ):

        scales = parameters[
            "scales"
        ].to(
            torch.float32
        )

        # Optimizer stores logarithmic scales.
        scales = torch.exp(
            scales
        )

        scales = torch.nan_to_num(
            scales,
            nan=1e-4,
            posinf=1e8,
            neginf=1e-8,
        )

        scales = torch.clamp(
            scales,
            min=1e-8,
            max=1e8,
        )

        return scales

    # =============================================================
    # ROTATIONS / QUATERNIONS
    # =============================================================

    def _prepare_quaternions(
        self,
        parameters,
    ):

        if "rotations" in parameters:

            quats = parameters[
                "rotations"
            ]

        elif "quats" in parameters:

            quats = parameters[
                "quats"
            ]

        else:

            raise KeyError(
                "Gaussian parameters contain neither "
                "'rotations' nor 'quats'."
            )

        quats = quats.to(
            torch.float32
        )

        if quats.ndim != 2:
            raise ValueError(
                "Gaussian quaternions must have "
                "shape [N, 4]."
            )

        if quats.shape[-1] != 4:
            raise ValueError(
                "Gaussian quaternions must have "
                "shape [N, 4]."
            )

        norm = torch.linalg.vector_norm(
            quats,
            dim=-1,
            keepdim=True,
        )

        if not torch.isfinite(
            norm
        ).all():

            raise RuntimeError(
                "Gaussian quaternion norms contain "
                "non-finite values."
            )

        quats = quats / torch.clamp(
            norm,
            min=1e-8,
        )

        return quats

    # =============================================================
    # OPACITY
    # =============================================================

    def _prepare_opacities(
        self,
        parameters,
    ):

        opacities = parameters[
            "opacities"
        ].to(
            torch.float32
        )

        if opacities.ndim == 2:

            if opacities.shape[-1] != 1:
                raise ValueError(
                    "Two-dimensional opacities must "
                    "have shape [N, 1]."
                )

            opacities = opacities[
                :,
                0,
            ]

        opacities = torch.sigmoid(
            opacities
        )

        opacities = torch.clamp(
            opacities,
            0.0,
            1.0,
        )

        return opacities

    # =============================================================
    # APPEARANCE
    # =============================================================

    def _prepare_appearance(
        self,
        parameters,
        active_sh_degree,
    ):

        # =========================================================
        # CURRENT PROJECT PATH: RGB COLORS
        # =========================================================

        if "colors" in parameters:

            colors = parameters[
                "colors"
            ].to(
                torch.float32
            )

            if colors.ndim != 2:
                raise ValueError(
                    "parameters['colors'] must have "
                    "shape [N, 3]. "
                    f"Received {tuple(colors.shape)}."
                )

            if colors.shape[-1] != 3:
                raise ValueError(
                    "Gaussian colors must have "
                    "exactly 3 channels."
                )

            gaussian_count = parameters[
                "means"
            ].shape[0]

            if colors.shape[0] != gaussian_count:
                raise ValueError(
                    "Gaussian colors contain a "
                    "different number of Gaussians."
                )

            if not torch.isfinite(
                colors
            ).all():

                raise RuntimeError(
                    "Gaussian colors contain "
                    "non-finite values."
                )

            # The optimizer stores colors in unconstrained
            # logit space.
            colors = torch.sigmoid(
                colors
            )

            colors = torch.clamp(
                colors,
                0.0,
                1.0,
            )

            # RGB mode.
            active_sh_degree = None

            return (
                colors,
                None,
                None,
            )

        # =========================================================
        # OPTIONAL SH PATH
        # =========================================================
        #
        # Kept for compatibility if the optimizer is changed
        # to SH in the future.
        #
        # gsplat expects:
        #
        #     [N, K, 3]
        #
        # when sh_degree is supplied.
        #
        # =========================================================

        if "shs" in parameters:

            shs = parameters[
                "shs"
            ].to(
                torch.float32
            )

            if shs.ndim != 3:
                raise ValueError(
                    "parameters['shs'] must have "
                    "shape [N, K, 3]. "
                    f"Received {tuple(shs.shape)}."
                )

            if shs.shape[-1] != 3:
                raise ValueError(
                    "SH parameters must have "
                    "three color channels."
                )

            gaussian_count = parameters[
                "means"
            ].shape[0]

            if shs.shape[0] != gaussian_count:
                raise ValueError(
                    "SH parameters contain a "
                    "different number of Gaussians."
                )

            if not torch.isfinite(
                shs
            ).all():

                raise RuntimeError(
                    "Gaussian SH parameters contain "
                    "non-finite values."
                )

            coefficient_count = int(
                shs.shape[-2]
            )

            max_degree = int(
                coefficient_count ** 0.5
            ) - 1

            active_sh_degree = max(
                0,
                int(
                    active_sh_degree
                ),
            )

            active_sh_degree = min(
                active_sh_degree,
                max_degree,
            )

            return (
                None,
                shs,
                active_sh_degree,
            )

        raise RuntimeError(
            "Gaussian parameters contain neither "
            "'colors' nor 'shs'."
        )

    # =============================================================
    # FRAME VALIDATION
    # =============================================================

    def _validate_frame(
        self,
        frame,
    ):

        required = (
            "fx",
            "fy",
            "cx",
            "cy",
            "width",
            "height",
            "rotation",
            "translation",
        )

        for key in required:

            if key not in frame:
                raise KeyError(
                    f"Frame is missing required "
                    f"field '{key}'."
                )

        width = int(
            frame["width"]
        )

        height = int(
            frame["height"]
        )

        if width <= 0:
            raise ValueError(
                f"Invalid frame width: {width}."
            )

        if height <= 0:
            raise ValueError(
                f"Invalid frame height: {height}."
            )

    # =============================================================
    # RENDER
    # =============================================================

    def render(
        self,
        parameters,
        frame,
        active_sh_degree=0,
    ):

        # =========================================================
        # VALIDATE
        # =========================================================

        self._validate_frame(
            frame
        )

        self._validate_parameters(
            parameters
        )

        # =========================================================
        # GAUSSIAN MEANS
        # =========================================================

        means = parameters[
            "means"
        ].to(
            torch.float32
        )

        if not means.is_cuda:
            means = means.to(
                self.device
            )

        # =========================================================
        # SCALES
        # =========================================================

        scales = self._prepare_scales(
            parameters
        )

        # =========================================================
        # ROTATIONS
        # =========================================================

        quats = self._prepare_quaternions(
            parameters
        )

        # =========================================================
        # OPACITY
        # =========================================================

        opacities = self._prepare_opacities(
            parameters
        )

        # =========================================================
        # APPEARANCE
        # =========================================================

        (
            colors,
            shs,
            active_sh_degree,
        ) = self._prepare_appearance(
            parameters,
            active_sh_degree,
        )

        # =========================================================
        # CAMERA
        # =========================================================

        viewmat = self._view_matrix(
            frame
        ).unsqueeze(
            0
        )

        K = self._camera_matrix(
            frame
        ).unsqueeze(
            0
        )

        # =========================================================
        # BACKGROUND
        # =========================================================
        #
        # IMPORTANT:
        #
        # For the current YAML the background is black:
        #
        #     [0, 0, 0]
        #
        # Therefore backgrounds=None.
        #
        # This completely avoids the gsplat 1.5.3
        # packed/background assertion that caused:
        #
        #     AssertionError:
        #     torch.Size([1, 3])
        #
        # =========================================================

        if self.has_custom_background:

            backgrounds = (
                self.background
                .view(
                    1,
                    3,
                )
            )

        else:

            backgrounds = None

        # =========================================================
        # RASTERIZATION ARGUMENTS
        # =========================================================

        raster_kwargs = {

            "means":
                means,

            "quats":
                quats,

            "scales":
                scales,

            "opacities":
                opacities,

            "viewmats":
                viewmat,

            "Ks":
                K,

            "width":
                int(
                    frame["width"]
                ),

            "height":
                int(
                    frame["height"]
                ),

            "near_plane":
                self.near_plane,

            "far_plane":
                self.far_plane,

            "radius_clip":
                self.radius_clip,

            "eps2d":
                self.eps2d,

            "packed":
                self.packed,

            "tile_size":
                self.tile_size,

            "render_mode":
                "RGB",

            "sparse_grad":
                self.sparse_grad,

            "absgrad":
                self.absgrad,

            "rasterize_mode":
                self.rasterize_mode,

            "channel_chunk":
                self.channel_chunk,

            "camera_model":
                self.camera_model,
        }

        # =========================================================
        # APPEARANCE
        # =========================================================

        if shs is not None:

            raster_kwargs[
                "colors"
            ] = shs

            raster_kwargs[
                "sh_degree"
            ] = int(
                active_sh_degree
            )

        else:

            raster_kwargs[
                "colors"
            ] = colors

            # IMPORTANT:
            #
            # When using ordinary RGB colors, sh_degree must
            # remain None.
            #
            raster_kwargs[
                "sh_degree"
            ] = None

        # =========================================================
        # BACKGROUND
        # =========================================================

        if backgrounds is not None:

            raster_kwargs[
                "backgrounds"
            ] = backgrounds

        # =========================================================
        # API SAFETY
        # =========================================================
        #
        # Only pass arguments that actually exist in the installed
        # gsplat rasterization API.
        #
        # This protects the project against harmless API additions
        # or removals without silently changing required inputs.
        #
        # =========================================================

        raster_kwargs = {
            key: value
            for key, value in raster_kwargs.items()
            if key in self._supported_arguments
        }

        # =========================================================
        # RASTERIZATION
        # =========================================================

        (
            render_colors,
            render_alphas,
            info,
        ) = rasterization(
            **raster_kwargs
        )

        # =========================================================
        # OUTPUT VALIDATION
        # =========================================================

        if render_colors is None:

            raise RuntimeError(
                "gsplat returned no rendered colors."
            )

        if render_alphas is None:

            raise RuntimeError(
                "gsplat returned no rendered alpha."
            )

        if info is None:

            raise RuntimeError(
                "gsplat rasterization returned "
                "no metadata."
            )

        # =========================================================
        # MEANS2D
        # =========================================================

        if "means2d" not in info:

            raise RuntimeError(
                "gsplat rasterization did not return "
                "'means2d'. "
                "This is required by the Gaussian "
                "densification strategy."
            )

        means2d = info[
            "means2d"
        ]

        if means2d is None:

            raise RuntimeError(
                "gsplat returned "
                "info['means2d'] = None."
            )


        # =========================================================
        # IMAGE
        # =========================================================

        rendered_image = render_colors[
            0
        ]

        rendered_alpha = render_alphas[
            0
        ]

        # =========================================================
        # IMAGE SHAPE
        # =========================================================

        if rendered_image.ndim != 3:

            raise RuntimeError(
                "gsplat rendered image must have "
                "shape [H, W, C]. "
                f"Received "
                f"{tuple(rendered_image.shape)}."
            )

        if rendered_image.shape[-1] != 3:

            raise RuntimeError(
                "gsplat rendered image must contain "
                "exactly 3 color channels. "
                f"Received "
                f"{rendered_image.shape[-1]}."
            )

        # =========================================================
        # ALPHA SHAPE
        # =========================================================

        if rendered_alpha.ndim != 3:

            raise RuntimeError(
                "gsplat rendered alpha must have "
                "shape [H, W, 1]. "
                f"Received "
                f"{tuple(rendered_alpha.shape)}."
            )

        if rendered_alpha.shape[-1] != 1:

            raise RuntimeError(
                "gsplat rendered alpha must contain "
                "exactly one channel."
            )

        # =========================================================
        # NUMERICAL VALIDATION
        # =========================================================

        if not torch.isfinite(
            rendered_image
        ).all():

            raise RuntimeError(
                "Gaussian renderer produced "
                "non-finite RGB values."
            )

        if not torch.isfinite(
            rendered_alpha
        ).all():

            raise RuntimeError(
                "Gaussian renderer produced "
                "non-finite alpha values."
            )

        # =========================================================
        # CLAMP OUTPUT
        # =========================================================

        rendered_image = torch.clamp(
            rendered_image,
            0.0,
            1.0,
        )

        rendered_alpha = torch.clamp(
            rendered_alpha,
            0.0,
            1.0,
        )

        # =========================================================
        # FINAL RESULT
        # =========================================================
        #
        # Keep gsplat metadata at the TOP LEVEL.
        #
        # Trainer / DefaultStrategy expects:
        #
        #     info["means2d"]
        #
        # and AbsGS expects:
        #
        #     info["means2d"].absgrad
        #
        # =========================================================

        result = {
            "image":
                rendered_image,

            "alpha":
                rendered_alpha,

            "active_sh_degree":
                active_sh_degree,

            "background":
                self.background,

            **info,
        }

        return result