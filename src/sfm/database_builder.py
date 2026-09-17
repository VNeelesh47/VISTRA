
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm


class DatabaseBuilder:

    def __init__(
        self,
        database_path,
        records,
    ):
        self.database_path = Path(
            database_path
        )

        self.records = records

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =============================================================
    # BUILD DATABASE
    # =============================================================

    def build(
        self,
        camera,
        matches,
    ):

        # ---------------------------------------------------------
        # Remove old database
        # ---------------------------------------------------------

        if self.database_path.exists():

            self.database_path.unlink()

        image_ids = {}

        # ---------------------------------------------------------
        # CREATE DATABASE
        # ---------------------------------------------------------

        with pycolmap.Database.open(
            str(self.database_path)
        ) as database:

            # =====================================================
            # CAMERA
            # =====================================================

            camera_id = database.write_camera(
                camera
            )

            # =====================================================
            # IMAGES + KEYPOINTS
            # =====================================================

            for record in tqdm(
                self.records,
                desc="Writing images",
                unit="image",
            ):

                image = pycolmap.Image(
                    name=Path(
                        record["image_path"]
                    ).name,
                    camera_id=camera_id,
                )

                image_id = database.write_image(
                    image
                )

                image_ids[
                    record["frame_id"]
                ] = image_id

                # -------------------------------------------------
                # Load features
                # -------------------------------------------------

                features = np.load(
                    record["feature_path"]
                )

                keypoints = np.asarray(
                    features["keypoints"],
                    dtype=np.float32,
                )

                if (
                    keypoints.ndim != 2
                    or keypoints.shape[1] != 2
                ):
                    raise RuntimeError(
                        f"Invalid keypoints for "
                        f"frame {record['frame_id']}: "
                        f"{keypoints.shape}"
                    )

                database.write_keypoints(
                    image_id,
                    keypoints,
                )

            # =====================================================
            # MATCHES + GEOMETRY
            # =====================================================

            for match in tqdm(
                matches,
                desc="Writing matches",
                unit="pair",
            ):

                frame_a = int(
                    match["frame_a"]
                )

                frame_b = int(
                    match["frame_b"]
                )

                image_id1 = image_ids[
                    frame_a
                ]

                image_id2 = image_ids[
                    frame_b
                ]

                # -------------------------------------------------
                # RAW MATCHES
                # -------------------------------------------------

                raw_matches = np.asarray(
                    match["matches"],
                    dtype=np.uint32,
                )

                if (
                    raw_matches.ndim != 2
                    or raw_matches.shape[1] != 2
                ):
                    raise RuntimeError(
                        f"Invalid matches for "
                        f"{frame_a} -> {frame_b}: "
                        f"{raw_matches.shape}"
                    )

                database.write_matches(
                    image_id1,
                    image_id2,
                    raw_matches,
                )

                # -------------------------------------------------
                # TWO VIEW GEOMETRY
                # -------------------------------------------------

                geometry = match[
                    "geometry"
                ]

                # -------------------------------------------------
                # Extract inliers BEFORE writing
                # -------------------------------------------------

                inlier_matches = getattr(
                    geometry,
                    "inlier_matches",
                    None,
                )

                if inlier_matches is None:

                    raise RuntimeError(
                        f"Geometry contains no "
                        f"inlier_matches for "
                        f"{frame_a} -> {frame_b}"
                    )

                inlier_matches = np.asarray(
                    inlier_matches,
                    dtype=np.uint32,
                )

                print(
                    f"BEFORE DB WRITE: "
                    f"{frame_a} -> {frame_b} | "
                    f"raw={len(raw_matches)} | "
                    f"inliers={len(inlier_matches)} | "
                    f"config={getattr(geometry, 'config', None)}"
                )

                if len(inlier_matches) < 1:

                    raise RuntimeError(
                        f"Attempting to write geometry "
                        f"with ZERO inliers for "
                        f"{frame_a} -> {frame_b}"
                    )

                # -------------------------------------------------
                # IMPORTANT:
                # Write geometry
                # -------------------------------------------------

                database.write_two_view_geometry(
                    image_id1,
                    image_id2,
                    geometry,
                )

                # -------------------------------------------------
                # READ BACK IMMEDIATELY
                # -------------------------------------------------

                stored_geometry = (
                    database.read_two_view_geometry(
                        image_id1,
                        image_id2,
                    )
                )

                stored_inliers = getattr(
                    stored_geometry,
                    "inlier_matches",
                    None,
                )

                if stored_inliers is None:

                    raise RuntimeError(
                        f"DATABASE READBACK FAILED: "
                        f"{frame_a} -> {frame_b} "
                        f"returned None inlier_matches"
                    )

                stored_inliers = np.asarray(
                    stored_inliers,
                    dtype=np.uint32,
                )

                stored_config = getattr(
                    stored_geometry,
                    "config",
                    None,
                )

                print(
                    f"AFTER DB WRITE: "
                    f"{frame_a} -> {frame_b} | "
                    f"stored_inliers="
                    f"{len(stored_inliers)} | "
                    f"config="
                    f"{stored_config}"
                )

                # -------------------------------------------------
                # HARD VALIDATION
                # -------------------------------------------------

                if (
                    len(stored_inliers)
                    != len(inlier_matches)
                ):

                    raise RuntimeError(
                        "COLMAP DATABASE GEOMETRY "
                        "READBACK MISMATCH: "
                        f"{frame_a} -> {frame_b} | "
                        f"expected="
                        f"{len(inlier_matches)} | "
                        f"stored="
                        f"{len(stored_inliers)}"
                    )

        # =========================================================
        # DATABASE CLOSED
        # =========================================================

        print()
        print(
            "Database construction completed."
        )

        print(
            f"Database: "
            f"{self.database_path}"
        )

        # =========================================================
        # FINAL REOPEN TEST
        # =========================================================

        print()
        print(
            "Performing final database "
            "geometry verification..."
        )

        with pycolmap.Database.open(
            str(self.database_path)
        ) as database:

            pair_ids, inlier_counts = (
                database.read_two_view_geometry_num_inliers()
            )

            print(
                f"Stored geometry pairs: "
                f"{len(pair_ids)}"
            )

            print(
                f"Stored inlier-count entries: "
                f"{len(inlier_counts)}"
            )

            if len(inlier_counts) > 0:

                print(
                    f"Maximum stored inliers: "
                    f"{max(inlier_counts)}"
                )

                print(
                    f"Minimum stored inliers: "
                    f"{min(inlier_counts)}"
                )

                print(
                    f"Total stored inliers: "
                    f"{sum(inlier_counts)}"
                )

            strong_pairs = sum(
                1
                for count in inlier_counts
                if count >= 20
            )

            print(
                f"Pairs with >=20 inliers: "
                f"{strong_pairs}"
            )

            if strong_pairs == 0:

                raise RuntimeError(
                    "DATABASE BUILD FAILED: "
                    "COLMAP database contains no "
                    "two-view geometry with >=20 "
                    "stored inliers."
                )

        return self.database_path

