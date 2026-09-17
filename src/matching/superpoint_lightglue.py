from pathlib import Path

import cv2
import torch
from lightglue import LightGlue, SuperPoint


class SuperPointLightGlue:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")

        matching = config["matching"]

        self.max_num_keypoints = matching[
            "max_num_keypoints"
        ]

        self.filter_threshold = matching[
            "filter_threshold"
        ]

        self.superpoint = SuperPoint(
            max_num_keypoints=self.max_num_keypoints
        ).eval().to(self.device)

        self.lightglue = LightGlue(
            features="superpoint",
            filter_threshold=self.filter_threshold,
        ).eval().to(self.device)

        self.feature_cache = {}

    def _load_image(self, image_path):
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        image = torch.from_numpy(
            image
        ).float()

        image /= 255.0

        image = image.unsqueeze(0).unsqueeze(0)

        return image.to(self.device)

    @torch.inference_mode()
    def extract(self, image_path):
        image_path = Path(image_path)
        cache_key = str(image_path.resolve())

        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        image = self._load_image(
            image_path
        )

        features = self.superpoint.extract(
            image
        )

        self.feature_cache[cache_key] = {
            key: value
            for key, value in features.items()
        }

        return self.feature_cache[cache_key]

    @torch.inference_mode()
    def match(
        self,
        image0_path,
        image1_path,
    ):
        features0 = self.extract(
            image0_path
        )

        features1 = self.extract(
            image1_path
        )

        result = self.lightglue(
            {
                "image0": features0,
                "image1": features1,
            }
        )

        matches = result["matches"]

        if matches.ndim == 3:
            matches = matches[0]

        matches = matches.detach().cpu()

        keypoints0 = features0[
            "keypoints"
        ][0].detach().cpu()

        keypoints1 = features1[
            "keypoints"
        ][0].detach().cpu()

        if matches.numel() == 0:
            matches = torch.empty(
                (0, 2),
                dtype=torch.int64,
            )

        else:
            matches = matches.long()

        return {
            "image0": str(
                image0_path
            ),
            "image1": str(
                image1_path
            ),
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "matches": matches,
        }

    def clear_cache(self):
        self.feature_cache.clear()

        torch.cuda.empty_cache()