import cv2
import numpy as np


class GeometricVerifier:
    def __init__(self, config):
        self.threshold = config["matching"]["ransac_threshold"]
        self.confidence = config["matching"]["ransac_confidence"]
        self.min_inliers = config["matching"]["min_inliers"]

    def verify(self, keypoints_a, keypoints_b, matches):
        if len(matches) < 8:
            return {
                "verified": False,
                "inliers": np.empty(0, dtype=np.int32),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
            }

        points_a = keypoints_a[matches[:, 0]]
        points_b = keypoints_b[matches[:, 1]]

        _, mask = cv2.findFundamentalMat(
            points_a,
            points_b,
            cv2.FM_RANSAC,
            self.threshold,
            self.confidence,
        )

        if mask is None:
            return {
                "verified": False,
                "inliers": np.empty(0, dtype=np.int32),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
            }

        mask = mask.ravel().astype(bool)
        inliers = np.flatnonzero(mask)
        inlier_count = len(inliers)
        inlier_ratio = inlier_count / len(matches)

        return {
            "verified": inlier_count >= self.min_inliers,
            "inliers": inliers,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
        }