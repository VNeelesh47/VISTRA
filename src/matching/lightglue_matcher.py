from pathlib import Path

import gc
import numpy as np
import torch

from lightglue import LightGlue


class LightGlueMatcher:

    def __init__(self, config):

        self.config = config

        # =========================================================
        # DEVICE
        # =========================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        matching_config = config["matching"]

        # =========================================================
        # LIGHTGLUE SETTINGS
        # =========================================================

        self.filter_threshold = float(
            matching_config.get(
                "filter_threshold",
                0.05,
            )
        )

        self.depth_confidence = float(
            matching_config.get(
                "depth_confidence",
                0.90,
            )
        )

        self.width_confidence = float(
            matching_config.get(
                "width_confidence",
                0.95,
            )
        )

        # =========================================================
        # MODEL
        # =========================================================

        self.model = LightGlue(
            features="superpoint",

            depth_confidence=self.depth_confidence,

            width_confidence=self.width_confidence,

            filter_threshold=self.filter_threshold,

        ).eval().to(
            self.device
        )

        # =========================================================
        # FEATURE CACHE
        #
        # Keep only loaded features.
        # This improves speed for repeated image pairs.
        # =========================================================

        self._cache = {}

    # =============================================================
    # LOAD FEATURES
    # =============================================================

    def _load(self, path):

        path = Path(path)

        cache_key = str(
            path.resolve()
        )

        if cache_key in self._cache:

            return self._cache[
                cache_key
            ]

        data = np.load(
            path
        )

        # ---------------------------------------------------------
        # KEYPOINTS
        # ---------------------------------------------------------

        keypoints = np.asarray(
            data["keypoints"],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # DESCRIPTORS
        # ---------------------------------------------------------

        descriptors = np.asarray(
            data["descriptors"],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # IMAGE SIZE
        # ---------------------------------------------------------

        image_size = np.asarray(
            data["image_size"],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        if keypoints.ndim != 2:

            raise RuntimeError(
                f"Invalid keypoints shape in {path}: "
                f"{keypoints.shape}"
            )

        if keypoints.shape[1] != 2:

            raise RuntimeError(
                f"Keypoints must have shape [N,2], "
                f"got {keypoints.shape} in {path}"
            )

        if descriptors.ndim != 2:

            raise RuntimeError(
                f"Invalid descriptor shape in {path}: "
                f"{descriptors.shape}"
            )

        if descriptors.shape[0] != keypoints.shape[0]:

            raise RuntimeError(
                f"Keypoint/descriptor count mismatch in {path}: "
                f"{keypoints.shape[0]} keypoints vs "
                f"{descriptors.shape[0]} descriptors"
            )

        if image_size.size != 2:

            raise RuntimeError(
                f"Invalid image_size in {path}: "
                f"{image_size}"
            )

        if not np.isfinite(
            keypoints
        ).all():

            raise RuntimeError(
                f"Non-finite keypoints found in {path}"
            )

        if not np.isfinite(
            descriptors
        ).all():

            raise RuntimeError(
                f"Non-finite descriptors found in {path}"
            )

        # ---------------------------------------------------------
        # LIGHTGLUE EXPECTS BATCH DIMENSION
        # ---------------------------------------------------------

        keypoints = torch.from_numpy(
            keypoints
        ).unsqueeze(
            0
        )

        descriptors = torch.from_numpy(
            descriptors
        ).unsqueeze(
            0
        )

        image_size = torch.from_numpy(
            image_size
        ).reshape(
            1,
            2,
        )

        # ---------------------------------------------------------
        # MOVE TO GPU
        # ---------------------------------------------------------

        keypoints = keypoints.to(
            self.device,
            non_blocking=True,
        )

        descriptors = descriptors.to(
            self.device,
            non_blocking=True,
        )

        image_size = image_size.to(
            self.device,
            non_blocking=True,
        )

        # ---------------------------------------------------------
        # NORMALIZE DESCRIPTOR FORMAT
        # ---------------------------------------------------------

        descriptors = descriptors.contiguous()

        keypoints = keypoints.contiguous()

        image_size = image_size.contiguous()

        features = {

            "keypoints": keypoints,

            "descriptors": descriptors,

            "image_size": image_size,

        }

        self._cache[
            cache_key
        ] = features

        return features

    # =============================================================
    # CACHE CONTROL
    # =============================================================

    def clear_cache(self):

        self._cache.clear()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        gc.collect()

    # =============================================================
    # MATCH
    # =============================================================

    def match(
        self,
        feature_a,
        feature_b,
    ):

        data_a = self._load(
            feature_a
        )

        data_b = self._load(
            feature_b
        )

        # =========================================================
        # LIGHTGLUE
        # =========================================================

        try:

            with torch.inference_mode():

                result = self.model(
                    {
                        "image0": data_a,
                        "image1": data_b,
                    }
                )

        except RuntimeError as exc:

            # -----------------------------------------------------
            # CUDA OOM RECOVERY
            # -----------------------------------------------------

            if (
                torch.cuda.is_available()
                and "out of memory" in str(exc).lower()
            ):

                torch.cuda.empty_cache()

                gc.collect()

                raise RuntimeError(
                    "CUDA out-of-memory during LightGlue "
                    "matching. Reduce feature count or "
                    "enable stronger pair filtering."
                ) from exc

            raise

        # =========================================================
        # EXTRACT MATCHES
        # =========================================================

        matches = result.get(
            "matches",
            None,
        )

        scores = result.get(
            "scores",
            None,
        )

        if matches is None:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        if scores is None:

            scores = torch.ones(
                (
                    matches.shape[-2],
                ),
                dtype=torch.float32,
                device=matches.device,
            )

        # =========================================================
        # REMOVE BATCH DIMENSION
        # =========================================================

        if isinstance(
            matches,
            list,
        ):

            matches = matches[0]

        if isinstance(
            scores,
            list,
        ):

            scores = scores[0]

        if matches.ndim == 3:

            matches = matches[0]

        if scores.ndim == 2:

            scores = scores[0]

        # =========================================================
        # VALIDATE
        # =========================================================

        if matches.numel() == 0:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        # =========================================================
        # MOVE TO CPU
        # =========================================================

        matches = (
            matches
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        # =========================================================
        # FORMAT
        # =========================================================

        matches = np.asarray(
            matches,
            dtype=np.int64,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        )

        # =========================================================
        # MATCH SHAPE
        # =========================================================

        if matches.ndim != 2:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        if matches.shape[1] != 2:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        # =========================================================
        # SCORE SHAPE
        # =========================================================

        scores = scores.reshape(
            -1
        )

        if len(scores) != len(matches):

            # Some LightGlue versions can return
            # unexpected score formatting.

            scores = np.ones(
                len(matches),
                dtype=np.float32,
            )

        # =========================================================
        # VALIDITY FILTER
        # =========================================================

        valid = np.isfinite(
            scores
        )

        valid &= (
            matches[:, 0] >= 0
        )

        valid &= (
            matches[:, 1] >= 0
        )

        matches = matches[
            valid
        ]

        scores = scores[
            valid
        ]

        if len(matches) == 0:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
                ),
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        # =========================================================
        # ONE-TO-ONE CORRESPONDENCE SAFETY
        # =========================================================

        # LightGlue normally already provides unique
        # correspondences. Keep the highest-scoring
        # correspondence if duplicates occur.

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

            used_a.add(
                a
            )

            used_b.add(
                b
            )

            selected.append(
                index
            )

        if not selected:

            return (
                np.empty(
                    (0, 2),
                    dtype=np.uint32,
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

        matches = matches[
            selected
        ]

        scores = scores[
            selected
        ]

        # =========================================================
        # FINAL FORMAT
        # =========================================================

        matches = np.ascontiguousarray(
            matches,
            dtype=np.uint32,
        )

        scores = np.ascontiguousarray(
            scores,
            dtype=np.float32,
        )

        return matches, scores