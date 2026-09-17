import numpy as np
import pycolmap


class GeometryEstimator:

    def __init__(self, config):

        matching = config["matching"]

        # =========================================================
        # TWO-VIEW GEOMETRY OPTIONS
        # =========================================================

        options = pycolmap.TwoViewGeometryOptions()

        # ---------------------------------------------------------
        # Minimum inliers required by COLMAP during estimation.
        #
        # Keep this relatively relaxed here.
        # Stronger filtering is performed after estimation.
        # ---------------------------------------------------------

        options.min_num_inliers = int(
            matching.get(
                "geometry_min_inliers",
                8,
            )
        )

        # =========================================================
        # RANSAC
        # =========================================================

        options.ransac.max_error = float(
            matching.get(
                "ransac_threshold",
                2.0,
            )
        )

        options.ransac.confidence = float(
            matching.get(
                "ransac_confidence",
                0.999,
            )
        )

        options.ransac.max_num_trials = int(
            matching.get(
                "ransac_max_trials",
                20000,
            )
        )

        # ---------------------------------------------------------
        # IMPORTANT
        #
        # estimate() uses self.options.
        # Store the configured options on the object.
        # ---------------------------------------------------------

        self.options = options

        # =========================================================
        # QUALITY SETTINGS
        # =========================================================

        self.min_inliers = int(
            matching.get(
                "min_inliers",
                8,
            )
        )

        self.min_inlier_ratio = float(
            matching.get(
                "min_inlier_ratio",
                0.05,
            )
        )

        self.min_matches = int(
            matching.get(
                "min_matches",
                20,
            )
        )

    # =============================================================
    # INPUT VALIDATION
    # =============================================================

    @staticmethod
    def _validate_keypoints(keypoints):

        try:
            keypoints = np.asarray(
                keypoints,
                dtype=np.float64,
            )
        except Exception:
            return None

        # Must be N x 2
        if keypoints.ndim != 2:
            return None

        if keypoints.shape[1] != 2:
            return None

        # At least 8 points are needed for two-view geometry.
        if len(keypoints) < 8:
            return None

        # Reject NaN / Inf.
        if not np.isfinite(keypoints).all():
            return None

        return np.ascontiguousarray(
            keypoints,
            dtype=np.float64,
        )

    # =============================================================
    # MATCH VALIDATION
    # =============================================================

    @staticmethod
    def _validate_matches(
        matches,
        keypoints1,
        keypoints2,
    ):

        try:
            matches = np.asarray(
                matches,
                dtype=np.int64,
            )
        except Exception:
            return None

        # Must be N x 2.
        if matches.ndim != 2:
            return None

        if matches.shape[1] != 2:
            return None

        # Need enough correspondences for geometry.
        if len(matches) < 8:
            return None

        # Reject NaN / Inf just in case the source array
        # is floating point.
        if not np.isfinite(matches).all():
            return None

        # Match indices cannot be negative.
        if np.any(matches < 0):
            return None

        # Validate keypoint indices.
        if (
            matches[:, 0].max()
            >= len(keypoints1)
        ):
            return None

        if (
            matches[:, 1].max()
            >= len(keypoints2)
        ):
            return None

        return np.ascontiguousarray(
            matches,
            dtype=np.uint32,
        )

    # =============================================================
    # REMOVE DUPLICATE MATCHES
    # =============================================================

    @staticmethod
    def _remove_duplicate_matches(matches):

        if len(matches) == 0:
            return matches

        used1 = set()
        used2 = set()

        selected = []

        for i in range(len(matches)):

            point1 = int(
                matches[i, 0]
            )

            point2 = int(
                matches[i, 1]
            )

            # One keypoint should correspond to only one
            # keypoint in the other image.
            if point1 in used1:
                continue

            if point2 in used2:
                continue

            used1.add(point1)
            used2.add(point2)

            selected.append(i)

        if not selected:

            return np.empty(
                (0, 2),
                dtype=np.uint32,
            )

        selected = np.asarray(
            selected,
            dtype=np.int64,
        )

        return np.ascontiguousarray(
            matches[selected],
            dtype=np.uint32,
        )

    # =============================================================
    # GET INLIERS FROM COLMAP
    # =============================================================

    @staticmethod
    def _get_geometry_inliers(geometry):

        inlier_matches = getattr(
            geometry,
            "inlier_matches",
            None,
        )

        if inlier_matches is None:
            return None

        try:
            inlier_matches = np.asarray(
                inlier_matches,
                dtype=np.uint32,
            )
        except Exception:
            return None

        if inlier_matches.ndim != 2:
            return None

        if inlier_matches.shape[1] != 2:
            return None

        if len(inlier_matches) == 0:
            return None

        return np.ascontiguousarray(
            inlier_matches,
            dtype=np.uint32,
        )

    # =============================================================
    # GEOMETRY TYPE
    # =============================================================

    @staticmethod
    def _geometry_type(geometry):

        geometry_type = getattr(
            geometry,
            "config",
            None,
        )

        if geometry_type is None:

            geometry_type = getattr(
                geometry,
                "type",
                None,
            )

        if geometry_type is None:
            return "UNKNOWN"

        return str(
            geometry_type
        )

    # =============================================================
    # ESTIMATE TWO-VIEW GEOMETRY
    # =============================================================

    def estimate(
        self,
        keypoints1,
        keypoints2,
        camera1,
        camera2,
        matches,
    ):

        # =========================================================
        # VALIDATE KEYPOINTS
        # =========================================================

        keypoints1 = (
            self._validate_keypoints(
                keypoints1
            )
        )

        keypoints2 = (
            self._validate_keypoints(
                keypoints2
            )
        )

        if keypoints1 is None:
            return None

        if keypoints2 is None:
            return None

        # =========================================================
        # VALIDATE MATCHES
        # =========================================================

        matches = (
            self._validate_matches(
                matches,
                keypoints1,
                keypoints2,
            )
        )

        if matches is None:
            return None

        # =========================================================
        # MINIMUM RAW MATCHES
        # =========================================================

        if len(matches) < self.min_matches:
            return None

        # =========================================================
        # REMOVE DUPLICATE CORRESPONDENCES
        # =========================================================

        matches = (
            self._remove_duplicate_matches(
                matches
            )
        )

        if len(matches) < self.min_matches:
            return None

        # =========================================================
        # TWO-VIEW GEOMETRIC VERIFICATION
        # =========================================================

        try:

            geometry = (
                pycolmap.estimate_two_view_geometry(
                    camera1,
                    keypoints1,
                    camera2,
                    keypoints2,
                    matches,
                    self.options,
                )
            )

        except Exception as exc:

            print(
                "Two-view geometry failed:",
                repr(exc),
            )

            return None

        # =========================================================
        # CHECK GEOMETRY RESULT
        # =========================================================

        if geometry is None:
            return None

        # =========================================================
        # EXTRACT INLIER MATCHES
        # =========================================================

        inlier_matches = (
            self._get_geometry_inliers(
                geometry
            )
        )

        if inlier_matches is None:
            return None

        inlier_count = len(
            inlier_matches
        )

        # =========================================================
        # MINIMUM INLIERS
        # =========================================================

        if (
            inlier_count
            < self.min_inliers
        ):
            return None

        # =========================================================
        # INLIER RATIO
        # =========================================================

        inlier_ratio = (
            inlier_count
            / max(
                float(len(matches)),
                1.0,
            )
        )

        if (
            inlier_ratio
            < self.min_inlier_ratio
        ):
            return None

        # =========================================================
        # VALIDATE INLIER INDICES
        # =========================================================

        if len(inlier_matches) == 0:
            return None

        if (
            inlier_matches[:, 0].max()
            >= len(keypoints1)
        ):
            return None

        if (
            inlier_matches[:, 1].max()
            >= len(keypoints2)
        ):
            return None

        # =========================================================
        # STORE QUALITY INFORMATION
        # =========================================================

        try:

            geometry.inlier_count = int(
                inlier_count
            )

        except Exception:
            pass

        try:

            geometry.inlier_ratio = float(
                inlier_ratio
            )

        except Exception:
            pass

        try:

            geometry.geometry_type = (
                self._geometry_type(
                    geometry
                )
            )

        except Exception:
            pass

        # =========================================================
        # SUCCESS
        # =========================================================

        return geometry