from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import SuperPoint


class SuperPointExtractor:
    def __init__(self, config):
        self.device = torch.device("cuda")

        features_config = config["features"]

        self.max_keypoints = features_config[
            "max_keypoints"
        ]


        self.detection_threshold = float(
            features_config.get(
                "detection_threshold",
                0.0005,
            )
        )

        self.nms_radius = int(
            features_config.get(
                "nms_radius",
                4,
            )
        )


        self.max_image_size = features_config.get(
            "max_image_size",
            None,
        )

        if self.max_image_size is not None:
            self.max_image_size = int(
                self.max_image_size
            )

        self.model = SuperPoint(
            max_num_keypoints=self.max_keypoints,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius,
        ).eval().to(self.device)

    def extract(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        original_height, original_width = image.shape[:2]

        scale = 1.0

        if self.max_image_size is not None:
            longest_side = max(
                original_width,
                original_height,
            )

            if longest_side > self.max_image_size:
                scale = (
                    self.max_image_size
                    / longest_side
                )

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
            image.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            result = self.model({"image": tensor})

        keypoints = result["keypoints"][0].cpu().numpy()

        if scale != 1.0:
            keypoints = keypoints / scale

        return {
            "keypoints": keypoints,
            "descriptors": result["descriptors"][0].cpu().numpy(),
            "scores": result["keypoint_scores"][0].cpu().numpy(),
            "image_size": np.array(
                [original_width, original_height],
                dtype=np.int32,
            ),
        }


def save_features(features, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        keypoints=features["keypoints"],
        descriptors=features["descriptors"],
        scores=features["scores"],
        image_size=features["image_size"],
    )