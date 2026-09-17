import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class KeyframeSelector:
    def __init__(
        self,
        records,
        output_dir,
        config,
    ):
        self.records = records
        self.output_dir = Path(output_dir)

        self.config = config["selection"]

        self.min_interval = float(
            self.config["min_interval"]
        )

        self.max_interval = float(
            self.config["max_interval"]
        )

        self.min_difference = float(
            self.config["min_difference"]
        )

        self.min_frames = int(
            self.config.get(
                "min_frames",
                70,
            )
        )

    def _difference(
        self,
        first,
        second,
    ):
        first = cv2.resize(
            first,
            (320, 180),
            interpolation=cv2.INTER_AREA,
        )

        second = cv2.resize(
            second,
            (320, 180),
            interpolation=cv2.INTER_AREA,
        )

        first = cv2.cvtColor(
            first,
            cv2.COLOR_BGR2GRAY,
        )

        second = cv2.cvtColor(
            second,
            cv2.COLOR_BGR2GRAY,
        )

        return float(
            np.mean(
                cv2.absdiff(
                    first,
                    second,
                )
            )
        )

    def _load_image(
        self,
        record,
    ):
        path = record.get(
            "selected_path"
        )

        if not path:
            return None

        return cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

    def _normal_selection(
        self,
        records,
    ):
        selected = []

        last_image = None
        last_timestamp = None

        for record in records:
            image = self._load_image(
                record
            )

            if image is None:
                continue

            timestamp = float(
                record["timestamp"]
            )

            if last_image is None:
                keep = True

            else:
                time_gap = (
                    timestamp
                    - last_timestamp
                )

                difference = self._difference(
                    last_image,
                    image,
                )

                keep = (
                    time_gap >= self.min_interval
                    and (
                        difference
                        >= self.min_difference
                        or time_gap
                        >= self.max_interval
                    )
                )

            if keep:
                selected.append(
                    (
                        record,
                        image,
                        timestamp,
                    )
                )

                last_image = image
                last_timestamp = timestamp

        return selected

    def _minimum_frame_selection(
        self,
        records,
        already_selected,
    ):
        """
        Add additional spatially/temporally distributed frames
        until min_frames is reached.

        This is a fallback only. Normal selection is preferred.
        """

        selected_ids = {
            id(record)
            for record, _, _ in already_selected
        }

        remaining = [
            record
            for record in records
            if id(record) not in selected_ids
        ]

        if not remaining:
            return already_selected

        required = (
            self.min_frames
            - len(already_selected)
        )

        if required <= 0:
            return already_selected

        # Spread additional frames across the complete
        # available sequence instead of simply taking
        # consecutive frames.
        count = min(
            required,
            len(remaining),
        )

        if count == len(remaining):
            candidates = remaining

        else:
            indices = np.linspace(
                0,
                len(remaining) - 1,
                count,
                dtype=int,
            )

            indices = np.unique(
                indices
            )

            candidates = [
                remaining[index]
                for index in indices
            ]

        result = list(
            already_selected
        )

        for record in candidates:
            image = self._load_image(
                record
            )

            if image is None:
                continue

            timestamp = float(
                record["timestamp"]
            )

            result.append(
                (
                    record,
                    image,
                    timestamp,
                )
            )

            if len(result) >= self.min_frames:
                break

        return result

    def run(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # NORMAL QUALITY-AWARE SELECTION
        # --------------------------------------------------

        selected_entries = (
            self._normal_selection(
                self.records
            )
        )

        normal_count = len(
            selected_entries
        )

        # --------------------------------------------------
        # MINIMUM FRAME GUARANTEE
        # --------------------------------------------------

        if normal_count < self.min_frames:
            print()
            print(
                "Normal keyframe selection produced:",
                normal_count,
                "frames",
            )

            print(
                "Minimum requested:",
                self.min_frames,
            )

            print(
                "Adding additional usable frames..."
            )

            selected_entries = (
                self._minimum_frame_selection(
                    self.records,
                    selected_entries,
                )
            )

        # --------------------------------------------------
        # SORT BY ORIGINAL TIMESTAMP
        # --------------------------------------------------

        selected_entries.sort(
            key=lambda item:
            item[2]
        )

        # --------------------------------------------------
        # COPY + METADATA
        # --------------------------------------------------

        selected = []

        for (
            record,
            image,
            timestamp,
        ) in tqdm(
            selected_entries,
            desc="Writing keyframes",
            unit="frame",
        ):
            source = Path(
                record["selected_path"]
            )

            destination = (
                self.output_dir
                / source.name
            )

            if not destination.exists():
                shutil.copy2(
                    source,
                    destination,
                )

            record["keyframe"] = True

            record["keyframe_path"] = str(
                destination
            )

            selected.append(
                record
            )

        # --------------------------------------------------
        # METADATA
        # --------------------------------------------------

        metadata_path = (
            self.output_dir
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

        # --------------------------------------------------
        # REPORT
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("KEYFRAME SELECTION COMPLETE")
        print("=" * 70)

        print(
            f"Available filtered frames: "
            f"{len(self.records)}"
        )

        print(
            f"Normal selection:           "
            f"{normal_count}"
        )

        print(
            f"Minimum requested:           "
            f"{self.min_frames}"
        )

        print(
            f"Final selected frames:       "
            f"{len(selected)}"
        )

        if len(selected) < self.min_frames:
            print()
            print(
                "WARNING: Minimum frame count "
                "could not be reached."
            )

            print(
                "Not enough usable filtered "
                "frames were available."
            )

        else:
            print()
            print(
                "Minimum frame requirement satisfied."
            )

        print("=" * 70)

        return selected