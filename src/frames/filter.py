import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.frames.quality import (
    blur_score,
    perceptual_hash,
    quality_score,
)


class FrameFilter:
    """
    Quality filtering stage.

    Purpose:
        - Remove unreadable frames.
        - Remove genuinely blurry frames.
        - Remove very poor-quality frames.
        - Remove only near-identical consecutive frames.

    Temporal keyframe selection is handled separately.
    """

    def __init__(self, records, selected_dir, config):
        self.records = records
        self.selected_dir = Path(selected_dir)
        self.config = config

        filtering = config["filtering"]

        self.min_blur = float(
            filtering["min_blur_score"]
        )

        self.min_quality = float(
            filtering["min_quality_score"]
        )

        self.max_hash_distance = int(
            filtering["max_hash_distance"]
        )

    def run(self):
        self.selected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = []

        previous_hash = None

        rejected_blur = 0
        rejected_quality = 0
        rejected_duplicate = 0
        rejected_unreadable = 0

        for record in tqdm(
            self.records,
            desc="Filtering frames",
            unit="frame",
        ):
            image = cv2.imread(
                record["path"],
                cv2.IMREAD_COLOR,
            )

            if image is None:
                record["selected"] = False
                record["rejection_reason"] = "unreadable"

                rejected_unreadable += 1
                continue

            sharpness = float(
                blur_score(image)
            )

            quality = float(
                quality_score(image)
            )

            current_hash = perceptual_hash(
                image
            )

            record["sharpness"] = sharpness
            record["quality_score"] = quality

            # --------------------------------------------------
            # QUALITY CHECK
            # --------------------------------------------------

            if sharpness < self.min_blur:
                record["selected"] = False
                record["rejection_reason"] = "blur"

                rejected_blur += 1
                continue

            if quality < self.min_quality:
                record["selected"] = False
                record["rejection_reason"] = "quality"

                rejected_quality += 1
                continue

            # --------------------------------------------------
            # DUPLICATE CHECK
            # --------------------------------------------------

            is_duplicate = False

            if previous_hash is not None:
                hash_distance = (
                    current_hash - previous_hash
                )

                is_duplicate = (
                    hash_distance
                    <= self.max_hash_distance
                )

            if is_duplicate:
                record["selected"] = False
                record["rejection_reason"] = "duplicate"

                rejected_duplicate += 1
                continue

            # --------------------------------------------------
            # KEEP FRAME
            # --------------------------------------------------

            destination = (
                self.selected_dir
                / Path(record["path"]).name
            )

            shutil.copy2(
                record["path"],
                destination,
            )

            record["selected"] = True

            record["selected_path"] = str(
                destination
            )

            record["rejection_reason"] = None

            # ImageHash itself is not JSON serializable.
            # Store its hexadecimal string representation.
            record["hash"] = str(
                current_hash
            )

            previous_hash = current_hash

            selected.append(record)

        # ------------------------------------------------------
        # METADATA
        # ------------------------------------------------------

        metadata_path = (
            self.selected_dir
            / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                selected,
                file,
                indent=2,
            )

        # ------------------------------------------------------
        # REPORT
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("FRAME FILTERING COMPLETE")
        print("=" * 70)

        print(
            f"Input frames:        {len(self.records)}"
        )

        print(
            f"Selected frames:     {len(selected)}"
        )

        print(
            f"Rejected blur:       {rejected_blur}"
        )

        print(
            f"Rejected quality:    {rejected_quality}"
        )

        print(
            f"Rejected duplicate:  {rejected_duplicate}"
        )

        print(
            f"Rejected unreadable: {rejected_unreadable}"
        )

        if self.records:
            percentage = (
                len(selected)
                / len(self.records)
                * 100.0
            )
        else:
            percentage = 0.0

        print(
            f"Retention:           {percentage:.1f}%"
        )

        print("=" * 70)

        return selected