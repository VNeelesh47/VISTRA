from pathlib import Path

import pycolmap


class DatabaseValidator:

    def __init__(
        self,
        database_path,
        config,
    ):
        self.database_path = Path(
            database_path
        )

        self.config = config

    # =============================================================
    # PAIR ID -> IMAGE IDS
    # =============================================================

    @staticmethod
    def _pair_id_to_image_ids(
        pair_id,
    ):
        """
        Convert COLMAP pair_id into image IDs.

        COLMAP uses:

            pair_id =
                2147483647 * image_id1
                + image_id2

        with image_id1 < image_id2.
        """

        pair_id = int(
            pair_id
        )

        base = 2147483647

        image_id2 = (
            pair_id % base
        )

        image_id1 = (
            pair_id - image_id2
        ) // base

        return (
            int(image_id1),
            int(image_id2),
        )

    # =============================================================
    # MAIN VALIDATION
    # =============================================================

    def validate(self):

        if not self.database_path.exists():

            raise FileNotFoundError(
                f"Database not found: "
                f"{self.database_path}"
            )

        with pycolmap.Database.open(
            str(
                self.database_path
            )
        ) as database:

            # =====================================================
            # BASIC DATABASE STATISTICS
            # =====================================================

            cameras = (
                database.num_cameras()
            )

            images = (
                database.num_images()
            )

            keypoints = (
                database.num_keypoints()
            )

            matches = (
                database.num_matches()
            )

            verified_pairs = (
                database.num_verified_image_pairs()
            )

            # =====================================================
            # CONFIGURATION
            # =====================================================

            reconstruction_config = (
                self.config.get(
                    "reconstruction",
                    {},
                )
            )

            min_images = int(
                reconstruction_config.get(
                    "min_images",
                    3,
                )
            )

            min_keypoints = int(
                reconstruction_config.get(
                    "min_keypoints",
                    1000,
                )
            )

            min_matches = int(
                reconstruction_config.get(
                    "min_matches",
                    500,
                )
            )

            min_verified_pairs = int(
                reconstruction_config.get(
                    "min_verified_pairs",
                    2,
                )
            )

            selection_config = (
                self.config.get(
                    "selection",
                    {},
                )
            )

            expected_frames = int(
                selection_config.get(
                    "min_frames",
                    min_images,
                )
            )

            matching_config = (
                self.config.get(
                    "matching",
                    {},
                )
            )

            min_pair_matches = int(
                matching_config.get(
                    "min_matches",
                    30,
                )
            )

            min_pair_inliers = int(
                matching_config.get(
                    "min_inliers",
                    15,
                )
            )

            min_inlier_ratio = float(
                matching_config.get(
                    "min_inlier_ratio",
                    0.20,
                )
            )

            # =====================================================
            # VALIDATION CONTAINERS
            # =====================================================

            errors = []

            warnings = []

            # =====================================================
            # CAMERA VALIDATION
            # =====================================================

            if cameras == 0:

                errors.append(
                    "Database contains no cameras."
                )

            if cameras > 1:

                warnings.append(
                    "Database contains multiple cameras. "
                    "For a single drone camera, a shared camera "
                    "model is normally preferred."
                )

            # =====================================================
            # IMAGE VALIDATION
            # =====================================================

            if images < min_images:

                errors.append(
                    f"Insufficient images: "
                    f"{images} < {min_images}."
                )

            if expected_frames > 0:

                registered_input_ratio = (
                    images
                    / float(
                        expected_frames
                    )
                )

                if (
                    registered_input_ratio
                    < 0.30
                ):

                    warnings.append(
                        "The database contains less than 30% "
                        "of the expected keyframes. "
                        "This may lead to weak camera trajectory "
                        "coverage."
                    )

            # =====================================================
            # KEYPOINT VALIDATION
            # =====================================================

            if keypoints < min_keypoints:

                errors.append(
                    f"Insufficient keypoints: "
                    f"{keypoints} < {min_keypoints}."
                )

            if images > 0:

                keypoints_per_image = (
                    keypoints
                    / float(
                        images
                    )
                )

            else:

                keypoints_per_image = 0.0

            if (
                keypoints_per_image
                < 500
            ):

                warnings.append(
                    "Average keypoint count per image is low: "
                    f"{keypoints_per_image:.1f}."
                )

            # =====================================================
            # MATCH VALIDATION
            # =====================================================

            if matches < min_matches:

                errors.append(
                    f"Insufficient matches: "
                    f"{matches} < {min_matches}."
                )

            # =====================================================
            # VERIFIED PAIR VALIDATION
            # =====================================================

            if (
                verified_pairs
                < min_verified_pairs
            ):

                errors.append(
                    "Insufficient verified image pairs: "
                    f"{verified_pairs} < "
                    f"{min_verified_pairs}."
                )

            if images >= 3:

                minimum_connected_pairs = (
                    images - 1
                )

                if (
                    verified_pairs
                    < minimum_connected_pairs
                ):

                    warnings.append(
                        "Verified-pair graph may be weakly "
                        "connected: "
                        f"{verified_pairs} verified pairs for "
                        f"{images} images."
                    )

            # =====================================================
            # PAIR DENSITY
            # =====================================================

            if images > 1:

                possible_pairs = (
                    images
                    * (images - 1)
                ) // 2

                pair_density = (
                    verified_pairs
                    / float(
                        max(
                            possible_pairs,
                            1,
                        )
                    )
                )

            else:

                pair_density = 0.0

            if (
                images >= 10
                and pair_density < 0.01
            ):

                warnings.append(
                    "Very low verified-pair density: "
                    f"{pair_density:.4f}."
                )

            # =====================================================
            # DETAILED TWO-VIEW GEOMETRY VALIDATION
            # =====================================================

            weak_pairs = []

            strong_pairs = 0

            total_verified_inliers = 0

            stored_geometry_pairs = 0

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # pycolmap returns:
            #
            # (
            #     pair_ids,
            #     geometries
            # )
            #
            # NOT:
            #
            # [
            #     (image_id1, image_id2, geometry)
            # ]
            # -----------------------------------------------------

            try:

                pair_ids, geometries = (
                    database.read_two_view_geometries()
                )

            except Exception as exc:

                errors.append(
                    "Failed to read two-view geometries "
                    f"from database: {exc}"
                )

                pair_ids = []

                geometries = []

            # =====================================================
            # CHECK LENGTH
            # =====================================================

            if (
                len(pair_ids)
                != len(geometries)
            ):

                errors.append(
                    "Two-view geometry database is inconsistent: "
                    f"{len(pair_ids)} pair IDs but "
                    f"{len(geometries)} geometries."
                )

            # =====================================================
            # PROCESS GEOMETRIES
            # =====================================================

            geometry_count = min(
                len(pair_ids),
                len(geometries),
            )

            for index in range(
                geometry_count
            ):

                pair_id = pair_ids[
                    index
                ]

                geometry = geometries[
                    index
                ]

                # -------------------------------------------------
                # Decode COLMAP pair ID.
                # -------------------------------------------------

                try:

                    image_id1, image_id2 = (
                        self._pair_id_to_image_ids(
                            pair_id
                        )
                    )

                except Exception:

                    continue

                # -------------------------------------------------
                # Inlier matches.
                # -------------------------------------------------

                inlier_matches = getattr(
                    geometry,
                    "inlier_matches",
                    None,
                )

                if inlier_matches is None:

                    continue

                try:

                    inlier_count = len(
                        inlier_matches
                    )

                except TypeError:

                    continue

                stored_geometry_pairs += 1

                total_verified_inliers += (
                    inlier_count
                )

                # -------------------------------------------------
                # Strong pair.
                # -------------------------------------------------

                if (
                    inlier_count
                    >= min_pair_inliers
                ):

                    strong_pairs += 1

                # -------------------------------------------------
                # Weak pair.
                # -------------------------------------------------

                else:

                    weak_pairs.append(
                        {
                            "image_id1": (
                                image_id1
                            ),
                            "image_id2": (
                                image_id2
                            ),
                            "inliers": (
                                inlier_count
                            ),
                        }
                    )

            # =====================================================
            # STRONG PAIR VALIDATION
            # =====================================================

            if (
                verified_pairs > 0
                and strong_pairs == 0
            ):

                errors.append(
                    "No verified image pair contains enough "
                    "geometric inliers."
                )

            # =====================================================
            # LARGE FRACTION OF WEAK PAIRS
            # =====================================================

            if (
                verified_pairs >= 5
                and strong_pairs
                < max(
                    2,
                    int(
                        verified_pairs
                        * 0.25
                    ),
                )
            ):

                warnings.append(
                    "A large fraction of verified pairs "
                    "have weak geometric support."
                )

            # =====================================================
            # INLIER STATISTICS
            # =====================================================

            if stored_geometry_pairs > 0:

                mean_inliers = (
                    total_verified_inliers
                    / float(
                        stored_geometry_pairs
                    )
                )

            else:

                mean_inliers = 0.0

            if (
                stored_geometry_pairs > 0
                and mean_inliers
                < min_pair_inliers
            ):

                warnings.append(
                    "Mean verified inlier count is below "
                    "the configured pair-quality threshold: "
                    f"{mean_inliers:.1f} < "
                    f"{min_pair_inliers}."
                )

            # =====================================================
            # GEOMETRY / VERIFIED-PAIR CONSISTENCY
            # =====================================================

            if (
                stored_geometry_pairs
                != verified_pairs
            ):

                warnings.append(
                    "The number of stored two-view geometry "
                    "records differs from the reported verified "
                    "image-pair count: "
                    f"{stored_geometry_pairs} geometry records "
                    f"vs {verified_pairs} verified pairs."
                )

            # =====================================================
            # DRONE-SPECIFIC QUALITY WARNINGS
            # =====================================================

            if images >= 10:

                if (
                    verified_pairs
                    < images * 2
                ):

                    warnings.append(
                        "Low pair connectivity for drone "
                        "reconstruction. Increase temporal/global "
                        "matching coverage if possible."
                    )

                if (
                    keypoints_per_image
                    < 1500
                ):

                    warnings.append(
                        "Low average feature density for a drone "
                        "scene. Increasing useful SuperPoint "
                        "keypoints may improve reconstruction."
                    )

            # =====================================================
            # RESULT
            # =====================================================

            result = {

                "valid": not errors,

                "cameras": int(
                    cameras
                ),

                "images": int(
                    images
                ),

                "keypoints": int(
                    keypoints
                ),

                "matches": int(
                    matches
                ),

                "verified_pairs": int(
                    verified_pairs
                ),

                "keypoints_per_image": round(
                    keypoints_per_image,
                    3,
                ),

                "pair_density": round(
                    pair_density,
                    6,
                ),

                "mean_verified_inliers": round(
                    mean_inliers,
                    3,
                ),

                "strong_verified_pairs": int(
                    strong_pairs
                ),

                "weak_verified_pairs": int(
                    len(
                        weak_pairs
                    )
                ),

                "stored_geometry_pairs": int(
                    stored_geometry_pairs
                ),

                "total_verified_inliers": int(
                    total_verified_inliers
                ),

                "errors": errors,

                "warnings": warnings,
            }

            # =====================================================
            # PRINT DIAGNOSTICS
            # =====================================================

            print()

            print(
                "DATABASE QUALITY"
            )

            print(
                "-" * 70
            )

            print(
                "Cameras:",
                cameras,
            )

            print(
                "Images:",
                images,
            )

            print(
                "Keypoints:",
                keypoints,
            )

            print(
                "Keypoints / image:",
                f"{keypoints_per_image:.1f}",
            )

            print(
                "Matches:",
                matches,
            )

            print(
                "Verified pairs:",
                verified_pairs,
            )

            print(
                "Stored geometry pairs:",
                stored_geometry_pairs,
            )

            print(
                "Pair density:",
                f"{pair_density:.4f}",
            )

            print(
                "Mean verified inliers:",
                f"{mean_inliers:.1f}",
            )

            print(
                "Strong verified pairs:",
                strong_pairs,
            )

            print(
                "Weak verified pairs:",
                len(
                    weak_pairs
                ),
            )

            print(
                "Total verified inliers:",
                total_verified_inliers,
            )

            # =====================================================
            # WARNINGS
            # =====================================================

            if warnings:

                print()

                print(
                    "DATABASE WARNINGS"
                )

                print(
                    "-" * 70
                )

                for warning in warnings:

                    print(
                        "WARNING:",
                        warning,
                    )

            # =====================================================
            # HARD FAILURE
            # =====================================================

            if errors:

                raise RuntimeError(
                    "COLMAP database validation failed:\n"
                    + "\n".join(
                        errors
                    )
                )

            print()

            print(
                "Database validation: PASSED"
            )

            return result