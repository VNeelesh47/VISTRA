import json
from pathlib import Path

from tqdm import tqdm

from src.features.superpoint import (
    SuperPointExtractor,
    save_features,
)


class FeatureExtractionPipeline:
    def __init__(
        self,
        records,
        output_dir,
        config,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.extractor = SuperPointExtractor(
            config
        )

    def run(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        for record in tqdm(
            self.records,
            desc="Extracting SuperPoint features",
            unit="image",
        ):
            image_path = Path(
                record["keyframe_path"]
            )

            feature_path = (
                self.output_dir
                / f"{image_path.stem}.npz"
            )

            features = self.extractor.extract(
                image_path
            )

            save_features(
                features,
                feature_path,
            )

            results.append(
                {
                    "frame_id": record["frame_id"],
                    "image_path": str(
                        image_path
                    ),
                    "feature_path": str(
                        feature_path
                    ),
                    "keypoint_count": len(
                        features["keypoints"]
                    ),
                    "width": int(
                        features["image_size"][0]
                    ),
                    "height": int(
                        features["image_size"][1]
                    ),
                }
            )

        metadata_path = (
            self.output_dir / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
            )

        return results

