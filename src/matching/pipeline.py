from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.matching.global_pairs import GlobalPairGenerator
from src.matching.lightglue_matcher import LightGlueMatcher
from src.matching.pair_generator import PairGenerator
from src.sfm.database_builder import DatabaseBuilder
from src.sfm.geometry import GeometryEstimator


class MatchingPipeline:

    def __init__(
        self,
        records,
        feature_dir,
        database_path,
        camera,
        config,
    ):
        self.records = records

        self.feature_dir = Path(
            feature_dir
        )

        self.database_path = Path(
            database_path
        )

        self.camera = camera

        self.config = config

        # =========================================================
        # MATCHER
        # =========================================================

        self.matcher = LightGlueMatcher(
            config
        )

        # =========================================================
        # GEOMETRY ESTIMATOR
        # =========================================================

        self.geometry = GeometryEstimator(
            config
        )

        matching = config["matching"]

        # =========================================================
        # BASIC MATCH QUALITY
        # =========================================================

        self.min_matches = int(
            matching.get(
                "min_matches",
                12,
            )
        )

        self.min_inliers = int(
            matching.get(
                "min_inliers",
                5,
            )
        )

        self.min_inlier_ratio = float(
            matching.get(
                "min_inlier_ratio",
                0.05,
            )
        )

        # =========================================================
        # SPATIAL COVERAGE
        # =========================================================

        self.min_spatial_coverage = float(
            matching.get(
                "min_spatial_coverage",
                0.01,
            )
        )

        # =========================================================
        # LIGHTGLUE SCORE QUALITY
        # =========================================================

        self.min_match_score = float(
            matching.get(
                "min_match_score",
                0.05,
            )
        )

        self.min_median_match_score = float(
            matching.get(
                "min_median_match_score",
                0.05,
            )
        )

        # =========================================================
        # COVERAGE DISTRIBUTION
        # =========================================================

        self.min_axis_coverage = float(
            matching.get(
                "min_axis_coverage",
                0.02,
            )
        )

        # =========================================================
        # PAIR LIMITS
        # =========================================================

        self.temporal_window = int(
            matching.get(
                "temporal_window",
                20,
            )
        )

        self.global_candidates = int(
            matching.get(
                "global_candidates",
                12,
            )
        )

        # =========================================================
        # DRONE-SPECIFIC QUALITY
        # =========================================================

        self.max_match_concentration = float(
            matching.get(
                "max_match_concentration",
                0.98,
            )
        )

        # =========================================================
        # INTERNAL STATISTICS
        # =========================================================

        self.stats = {
            "pairs_tested": 0,
            "rejected_weak_matches": 0,
            "rejected_match_score": 0,
            "rejected_geometry": 0,
            "rejected_inlier_count": 0,
            "rejected_inlier_ratio": 0,
            "rejected_spatial_coverage": 0,
            "rejected_match_concentration": 0,
        }

    # =============================================================
    # FEATURE PATH
    # =============================================================

    def _feature_path(
        self,
        frame_id,
    ):
        return (
            self.feature_dir
            / f"{frame_id:06d}.npz"
        )

    # =============================================================
    # LOAD KEYPOINTS
    # =============================================================

    def _load_keypoints(
        self,
        frame_id,
    ):
        feature_path = (
            self._feature_path(
                frame_id
            )
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: "
                f"{feature_path}"
            )

        data = np.load(
            feature_path
        )

        if "keypoints" not in data:
            raise RuntimeError(
                f"Feature file does not contain "
                f"'keypoints': {feature_path}"
            )

        keypoints = np.asarray(
            data["keypoints"],
            dtype=np.float32,
        )

        if keypoints.ndim != 2:
            raise RuntimeError(
                f"Invalid keypoint shape in "
                f"{feature_path}: "
                f"{keypoints.shape}"
            )

        if keypoints.shape[1] != 2:
            raise RuntimeError(
                f"Expected Nx2 keypoints in "
                f"{feature_path}, got "
                f"{keypoints.shape}"
            )

        return np.ascontiguousarray(
            keypoints,
            dtype=np.float32,
        )

    # =============================================================
    # PAIR GENERATION
    # =============================================================

    def _generate_pairs(self):

        temporal_pairs = PairGenerator(
            self.records,
            self.config,
        ).generate()

        global_pairs = GlobalPairGenerator(
            self.records,
            self.config,
        ).generate()

        temporal_pairs = set(
            temporal_pairs
        )

        global_pairs = set(
            global_pairs
        )

        pairs = (
            temporal_pairs
            | global_pairs
        )

        # =========================================================
        # NORMALIZE PAIR ORDER
        # =========================================================

        normalized_pairs = set()

        for pair in pairs:

            if len(pair) != 2:
                continue

            frame_a = int(
                pair[0]
            )

            frame_b = int(
                pair[1]
            )

            if frame_a == frame_b:
                continue

            if frame_a > frame_b:
                frame_a, frame_b = (
                    frame_b,
                    frame_a,
                )

            normalized_pairs.add(
                (
                    frame_a,
                    frame_b,
                )
            )

        return sorted(
            normalized_pairs
        )

    # =============================================================
    # SPATIAL COVERAGE
    # =============================================================

    @staticmethod
    def _spatial_coverage(
        keypoints,
        matches,
        width,
        height,
    ):

        if len(matches) == 0:
            return 0.0

        points = keypoints[
            matches[:, 0]
        ]

        if len(points) < 3:
            return 0.0

        x_min = float(
            points[:, 0].min()
        )

        x_max = float(
            points[:, 0].max()
        )

        y_min = float(
            points[:, 1].min()
        )

        y_max = float(
            points[:, 1].max()
        )

        box_width = max(
            0.0,
            x_max - x_min,
        )

        box_height = max(
            0.0,
            y_max - y_min,
        )

        image_area = max(
            float(width * height),
            1.0,
        )

        coverage = (
            box_width
            * box_height
            / image_area
        )

        return float(
            coverage
        )

    # =============================================================
    # AXIS COVERAGE
    # =============================================================

    @staticmethod
    def _axis_coverage(
        keypoints,
        matches,
        width,
        height,
    ):

        if len(matches) == 0:
            return 0.0

        points = keypoints[
            matches[:, 0]
        ]

        if len(points) < 3:
            return 0.0

        x_range = (
            float(
                points[:, 0].max()
            )
            -
            float(
                points[:, 0].min()
            )
        )

        y_range = (
            float(
                points[:, 1].max()
            )
            -
            float(
                points[:, 1].min()
            )
        )

        normalized_x = (
            x_range
            /
            max(
                float(width),
                1.0,
            )
        )

        normalized_y = (
            y_range
            /
            max(
                float(height),
                1.0,
            )
        )

        return float(
            min(
                normalized_x,
                normalized_y,
            )
        )

    # =============================================================
    # MATCH CONCENTRATION
    # =============================================================

    @staticmethod
    def _match_concentration(
        keypoints,
        matches,
        width,
        height,
    ):

        if len(matches) == 0:
            return 1.0

        points = keypoints[
            matches[:, 0]
        ]

        if len(points) < 4:
            return 1.0

        x_min = float(
            points[:, 0].min()
        )

        x_max = float(
            points[:, 0].max()
        )

        y_min = float(
            points[:, 1].min()
        )

        y_max = float(
            points[:, 1].max()
        )

        box_width = max(
            0.0,
            x_max - x_min,
        )

        box_height = max(
            0.0,
            y_max - y_min,
        )

        box_area = (
            box_width
            *
            box_height
        )

        image_area = max(
            float(width * height),
            1.0,
        )

        concentration = (
            box_area
            /
            image_area
        )

        return float(
            concentration
        )

    # =============================================================
    # MATCH SCORE VALIDATION
    # =============================================================

    def _validate_match_scores(
        self,
        scores,
    ):

        if scores is None:
            return True

        scores = np.asarray(
            scores,
            dtype=np.float32,
        )

        if scores.size == 0:
            return False

        scores = scores[
            np.isfinite(scores)
        ]

        if scores.size == 0:
            return False

        mean_score = float(
            scores.mean()
        )

        median_score = float(
            np.median(scores)
        )

        if (
            mean_score
            <
            self.min_match_score
        ):
            return False

        if (
            median_score
            <
            self.min_median_match_score
        ):
            return False

        return True

    # =============================================================
    # MATCH SCORE STATISTICS
    # =============================================================

    @staticmethod
    def _score_statistics(
        scores,
    ):

        if scores is None:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        scores = np.asarray(
            scores,
            dtype=np.float32,
        )

        if scores.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "mean": float(
                np.mean(scores)
            ),
            "median": float(
                np.median(scores)
            ),
            "min": float(
                np.min(scores)
            ),
            "max": float(
                np.max(scores)
            ),
        }

    # =============================================================
    # MATCH INDEX VALIDATION
    # =============================================================

    @staticmethod
    def _validate_match_indices(
        matches,
        keypoints_a,
        keypoints_b,
    ):

        if matches is None:
            return False

        matches = np.asarray(
            matches
        )

        if matches.ndim != 2:
            return False

        if matches.shape[1] != 2:
            return False

        if len(matches) == 0:
            return False

        if not np.isfinite(
            matches
        ).all():
            return False

        matches = matches.astype(
            np.int64,
            copy=False,
        )

        if (
            matches[:, 0].min()
            < 0
        ):
            return False

        if (
            matches[:, 1].min()
            < 0
        ):
            return False

        if (
            matches[:, 0].max()
            >= len(keypoints_a)
        ):
            return False

        if (
            matches[:, 1].max()
            >= len(keypoints_b)
        ):
            return False

        return True

    # =============================================================
    # DUPLICATE MATCH REMOVAL
    # =============================================================

    @staticmethod
    def _remove_duplicate_matches(
        matches,
        scores,
    ):

        if len(matches) == 0:
            return (
                matches,
                scores,
            )

        matches = np.asarray(
            matches,
            dtype=np.int64,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        )

        if len(matches) != len(scores):
            count = min(
                len(matches),
                len(scores),
            )

            matches = matches[
                :count
            ]

            scores = scores[
                :count
            ]

        if len(matches) == 0:
            return (
                np.empty(
                    (0, 2),
                    dtype=np.int64,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        # =========================================================
        # LIGHTGLUE ALREADY PRODUCES MUTUAL MATCHES.
        #
        # We retain this safety filter because malformed feature
        # files or future matcher changes could introduce duplicates.
        # =========================================================

        order = np.argsort(
            -scores
        )

        used_a = set()
        used_b = set()

        selected = []

        for index in order:

            a = int(
                matches[index, 0]
            )

            b = int(
                matches[index, 1]
            )

            if a in used_a:
                continue

            if b in used_b:
                continue

            used_a.add(a)
            used_b.add(b)

            selected.append(
                int(index)
            )

        if not selected:
            return (
                np.empty(
                    (0, 2),
                    dtype=np.int64,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        selected = np.asarray(
            selected,
            dtype=np.int64,
        )

        return (
            np.ascontiguousarray(
                matches[selected],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                scores[selected],
                dtype=np.float32,
            ),
        )

    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(self):

        results = []

        pairs = (
            self._generate_pairs()
        )

        print(
            f"Candidate pairs: "
            f"{len(pairs)}"
        )

        self.stats[
            "pairs_tested"
        ] = len(pairs)

        # =========================================================
        # PAIR PROCESSING
        # =========================================================

        for frame_a, frame_b in tqdm(
            pairs,
            desc="LightGlue matching",
            unit="pair",
        ):

            # -----------------------------------------------------
            # FEATURE FILES
            # -----------------------------------------------------

            feature_a = (
                self._feature_path(
                    frame_a
                )
            )

            feature_b = (
                self._feature_path(
                    frame_b
                )
            )

            if not feature_a.exists():

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            if not feature_b.exists():

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            # -----------------------------------------------------
            # KEYPOINTS
            # -----------------------------------------------------

            try:

                keypoints_a = (
                    self._load_keypoints(
                        frame_a
                    )
                )

                keypoints_b = (
                    self._load_keypoints(
                        frame_b
                    )
                )

            except Exception as exc:

                print(
                    f"\nKeypoint loading failed "
                    f"for {frame_a} <-> {frame_b}: "
                    f"{repr(exc)}"
                )

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            # -----------------------------------------------------
            # LIGHTGLUE
            # -----------------------------------------------------

            try:

                matches, scores = (
                    self.matcher.match(
                        feature_a,
                        feature_b,
                    )
                )

            except Exception as exc:

                print(
                    f"\nLightGlue failed "
                    f"for {frame_a} <-> {frame_b}: "
                    f"{repr(exc)}"
                )

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            matches = np.asarray(
                matches
            )

            scores = np.asarray(
                scores,
                dtype=np.float32,
            )

            # =====================================================
            # DIAGNOSTIC INFORMATION
            # =====================================================

            if len(matches) > 0:

                mean_score = float(
                    np.mean(scores)
                )

                median_score = float(
                    np.median(scores)
                )

            else:

                mean_score = 0.0
                median_score = 0.0

            # -----------------------------------------------------
            # Print first 20 useful pair diagnostics.
            # -----------------------------------------------------

            diagnostic_count = (
                self.stats[
                    "rejected_weak_matches"
                ]
                +
                self.stats[
                    "rejected_match_score"
                ]
                +
                self.stats[
                    "rejected_geometry"
                ]
                +
                self.stats[
                    "rejected_inlier_count"
                ]
                +
                self.stats[
                    "rejected_inlier_ratio"
                ]
                +
                self.stats[
                    "rejected_spatial_coverage"
                ]
                +
                self.stats[
                    "rejected_match_concentration"
                ]
            )

            if diagnostic_count < 20:

                print(
                    f"\nPAIR "
                    f"{frame_a} <-> {frame_b} | "
                    f"raw_matches={len(matches)} | "
                    f"mean_score={mean_score:.4f} | "
                    f"median_score={median_score:.4f}"
                )

            # -----------------------------------------------------
            # VALIDATE MATCH INDICES
            # -----------------------------------------------------

            if not self._validate_match_indices(
                matches,
                keypoints_a,
                keypoints_b,
            ):

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            # -----------------------------------------------------
            # REMOVE DUPLICATE MATCHES
            # -----------------------------------------------------

            matches, scores = (
                self._remove_duplicate_matches(
                    matches,
                    scores,
                )
            )

            # -----------------------------------------------------
            # MINIMUM MATCH COUNT
            # -----------------------------------------------------

            if (
                len(matches)
                < self.min_matches
            ):

                self.stats[
                    "rejected_weak_matches"
                ] += 1

                continue

            # -----------------------------------------------------
            # MATCH CONFIDENCE
            # -----------------------------------------------------

            if not self._validate_match_scores(
                scores
            ):

                self.stats[
                    "rejected_match_score"
                ] += 1

                continue

            # =====================================================
            # GEOMETRIC VERIFICATION
            # =====================================================

            geometry = (
                self.geometry.estimate(
                    keypoints_a,
                    keypoints_b,
                    self.camera,
                    self.camera,
                    matches,
                )
            )

            if geometry is None:

                # -------------------------------------------------
                # IMPORTANT DIAGNOSTIC
                # -------------------------------------------------

                if diagnostic_count < 20:

                    print(
                        f"GEOMETRY FAILED "
                        f"{frame_a} <-> {frame_b} | "
                        f"matches={len(matches)}"
                    )

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            # -----------------------------------------------------
            # INLIER MATCHES
            # -----------------------------------------------------

            inlier_matches = getattr(
                geometry,
                "inlier_matches",
                None,
            )

            if inlier_matches is None:

                if diagnostic_count < 20:

                    print(
                        f"GEOMETRY RETURNED "
                        f"NO INLIERS "
                        f"{frame_a} <-> {frame_b}"
                    )

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            inlier_matches = np.asarray(
                inlier_matches,
                dtype=np.uint32,
            )

            if (
                inlier_matches.ndim != 2
                or
                inlier_matches.shape[1] != 2
            ):

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            inlier_count = len(
                inlier_matches
            )

            # -----------------------------------------------------
            # IMPORTANT DIAGNOSTIC
            # -----------------------------------------------------

            if diagnostic_count < 20:

                print(
                    f"GEOMETRY SUCCESS "
                    f"{frame_a} <-> {frame_b} | "
                    f"matches={len(matches)} | "
                    f"inliers={inlier_count}"
                )

            # -----------------------------------------------------
            # MINIMUM INLIER COUNT
            # -----------------------------------------------------

            if (
                inlier_count
                < self.min_inliers
            ):

                self.stats[
                    "rejected_inlier_count"
                ] += 1

                continue

            # -----------------------------------------------------
            # INLIER RATIO
            # -----------------------------------------------------

            inlier_ratio = (
                inlier_count
                /
                max(
                    float(len(matches)),
                    1.0,
                )
            )

            if (
                inlier_ratio
                < self.min_inlier_ratio
            ):

                self.stats[
                    "rejected_inlier_ratio"
                ] += 1

                continue

            # -----------------------------------------------------
            # VALIDATE INLIER INDICES
            # -----------------------------------------------------

            if (
                inlier_matches[:, 0].min()
                < 0
            ):

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            if (
                inlier_matches[:, 1].min()
                < 0
            ):

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            if (
                inlier_matches[:, 0].max()
                >= len(keypoints_a)
            ):

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            if (
                inlier_matches[:, 1].max()
                >= len(keypoints_b)
            ):

                self.stats[
                    "rejected_geometry"
                ] += 1

                continue

            # =====================================================
            # IMAGE DIMENSIONS
            # =====================================================

            width = int(
                self.camera.width
            )

            height = int(
                self.camera.height
            )

            # =====================================================
            # SPATIAL COVERAGE — IMAGE A
            # =====================================================

            coverage_a = (
                self._spatial_coverage(
                    keypoints_a,
                    inlier_matches,
                    width,
                    height,
                )
            )

            # =====================================================
            # SPATIAL COVERAGE — IMAGE B
            # =====================================================

            coverage_b = (
                self._spatial_coverage(
                    keypoints_b,
                    inlier_matches,
                    width,
                    height,
                )
            )

            coverage = min(
                coverage_a,
                coverage_b,
            )

            if (
                coverage
                < self.min_spatial_coverage
            ):

                self.stats[
                    "rejected_spatial_coverage"
                ] += 1

                continue

            # =====================================================
            # AXIS COVERAGE
            # =====================================================

            axis_coverage_a = (
                self._axis_coverage(
                    keypoints_a,
                    inlier_matches,
                    width,
                    height,
                )
            )

            axis_coverage_b = (
                self._axis_coverage(
                    keypoints_b,
                    inlier_matches,
                    width,
                    height,
                )
            )

            axis_coverage = min(
                axis_coverage_a,
                axis_coverage_b,
            )

            if (
                axis_coverage
                < self.min_axis_coverage
            ):

                self.stats[
                    "rejected_spatial_coverage"
                ] += 1

                continue

            # =====================================================
            # MATCH CONCENTRATION
            # =====================================================

            concentration_a = (
                self._match_concentration(
                    keypoints_a,
                    inlier_matches,
                    width,
                    height,
                )
            )

            concentration_b = (
                self._match_concentration(
                    keypoints_b,
                    inlier_matches,
                    width,
                    height,
                )
            )

            concentration = min(
                concentration_a,
                concentration_b,
            )

            if (
                concentration
                > self.max_match_concentration
            ):

                self.stats[
                    "rejected_match_concentration"
                ] += 1

                continue

            # =====================================================
            # SCORE STATISTICS
            # =====================================================

            score_statistics = (
                self._score_statistics(
                    scores
                )
            )

            # =====================================================
            # KEEP VERIFIED PAIR
            # =====================================================

            results.append(
                {
                    "frame_a": frame_a,

                    "frame_b": frame_b,

                    "matches": (
                        matches
                    ),

                    "scores": (
                        scores
                    ),

                    "geometry": (
                        geometry
                    ),

                    "inlier_matches": (
                        inlier_matches
                    ),

                    "inlier_count": (
                        int(
                            inlier_count
                        )
                    ),

                    "inlier_ratio": (
                        float(
                            inlier_ratio
                        )
                    ),

                    "spatial_coverage": (
                        float(
                            coverage
                        )
                    ),

                    "axis_coverage": (
                        float(
                            axis_coverage
                        )
                    ),

                    "match_concentration": (
                        float(
                            concentration
                        )
                    ),

                    "mean_match_score": (
                        score_statistics[
                            "mean"
                        ]
                    ),

                    "median_match_score": (
                        score_statistics[
                            "median"
                        ]
                    ),
                }
            )

        # =========================================================
        # FINAL STATISTICS
        # =========================================================

        print()
        print(
            "======================================================================"
        )
        print(
            "MATCHING SUMMARY"
        )
        print(
            "======================================================================"
        )

        print(
            f"Candidate pairs: "
            f"{len(pairs)}"
        )

        print(
            f"Verified pairs: "
            f"{len(results)}"
        )

        print(
            "Rejected weak matches:",
            self.stats[
                "rejected_weak_matches"
            ],
        )

        print(
            "Rejected match confidence:",
            self.stats[
                "rejected_match_score"
            ],
        )

        print(
            "Rejected geometry:",
            self.stats[
                "rejected_geometry"
            ],
        )

        print(
            "Rejected low inlier count:",
            self.stats[
                "rejected_inlier_count"
            ],
        )

        print(
            "Rejected low inlier ratio:",
            self.stats[
                "rejected_inlier_ratio"
            ],
        )

        print(
            "Rejected poor spatial coverage:",
            self.stats[
                "rejected_spatial_coverage"
            ],
        )

        print(
            "Rejected concentrated matches:",
            self.stats[
                "rejected_match_concentration"
            ],
        )

        # =========================================================
        # VERIFIED PAIR INFORMATION
        # =========================================================

        if results:

            print()
            print(
                "VERIFIED PAIRS"
            )

            for result in results[:20]:

                print(
                    f"{result['frame_a']} <-> "
                    f"{result['frame_b']} | "
                    f"matches="
                    f"{len(result['matches'])} | "
                    f"inliers="
                    f"{result['inlier_count']} | "
                    f"ratio="
                    f"{result['inlier_ratio']:.3f} | "
                    f"coverage="
                    f"{result['spatial_coverage']:.3f}"
                )

        # =========================================================
        # MINIMUM VERIFIED PAIRS
        # =========================================================

        min_verified_pairs = int(
            self.config[
                "reconstruction"
            ].get(
                "min_verified_pairs",
                2,
            )
        )

        if (
            len(results)
            < min_verified_pairs
        ):

            raise RuntimeError(
                "Too few geometrically reliable "
                "image pairs remain: "
                f"{len(results)} < "
                f"{min_verified_pairs}"
            )

        # =========================================================
        # DATABASE
        # =========================================================

        database = DatabaseBuilder(
            self.database_path,
            self.records,
        )

        database_path = (
            database.build(
                self.camera,
                results,
            )
        )

        # =========================================================
        # SUMMARY VALUES
        # =========================================================

        mean_inliers = float(
            np.mean(
                [
                    result[
                        "inlier_count"
                    ]
                    for result in results
                ]
            )
        )

        mean_inlier_ratio = float(
            np.mean(
                [
                    result[
                        "inlier_ratio"
                    ]
                    for result in results
                ]
            )
        )

        mean_coverage = float(
            np.mean(
                [
                    result[
                        "spatial_coverage"
                    ]
                    for result in results
                ]
            )
        )

        mean_match_score = float(
            np.mean(
                [
                    result[
                        "mean_match_score"
                    ]
                    for result in results
                ]
            )
        )

        return {
            "pairs_tested": (
                len(pairs)
            ),

            "pairs_verified": (
                len(results)
            ),

            "database": (
                str(
                    database_path
                )
            ),

            "mean_inliers": (
                mean_inliers
            ),

            "mean_inlier_ratio": (
                mean_inlier_ratio
            ),

            "mean_spatial_coverage": (
                mean_coverage
            ),

            "mean_match_score": (
                mean_match_score
            ),

            "matches": (
                results
            ),
        }