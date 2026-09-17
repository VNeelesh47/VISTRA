from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm


class GaussianInitializer:
    def __init__(self, config):
        self.config = config

    def _load_reconstruction(self, sparse_path):
        sparse_path = Path(sparse_path)

        if not sparse_path.exists():
            raise FileNotFoundError(
                f"Sparse reconstruction not found: {sparse_path}"
            )

        return pycolmap.Reconstruction(
            str(sparse_path)
        )

    def _extract_points(self, reconstruction):
        points = []

        iterator = tqdm(
            reconstruction.points3D.values(),
            desc="Extracting 3D points",
            unit="point",
        )

        for point3D in iterator:
            if point3D.xyz is None:
                continue

            if point3D.color is None:
                continue

            track = point3D.track

            try:
                track_length = track.length()
            except AttributeError:
                try:
                    track_length = len(track.elements)
                except AttributeError:
                    continue

            points.append(
                {
                    "xyz": np.asarray(
                        point3D.xyz,
                        dtype=np.float32,
                    ),
                    "color": np.asarray(
                        point3D.color,
                        dtype=np.float32,
                    ),
                    "error": float(point3D.error),
                    "track_length": int(track_length),
                }
            )

        if not points:
            raise RuntimeError(
                "Sparse reconstruction contains no valid 3D points."
            )

        return points

    def _filter_points(self, points):
        gaussian_config = self.config["gaussian"]

        min_track_length = int(
            gaussian_config["min_track_length"]
        )

        max_reprojection_error = float(
            gaussian_config["max_reprojection_error"]
        )

        filtered = [
            point
            for point in points
            if point["track_length"] >= min_track_length
            and point["error"] <= max_reprojection_error
        ]

        min_points = int(
            gaussian_config["min_points"]
        )

        if len(filtered) < min_points:
            raise RuntimeError(
                "Too few valid points remain after Gaussian "
                "initialization filtering: "
                f"{len(filtered)} < {min_points}"
            )

        return filtered

    def _estimate_scales(self, points):
        """
        Estimate one isotropic Gaussian radius per 3D point,
        based on distance to its nearest neighbor.

        Uses a KD-tree (O(N log N)) instead of a brute-force
        all-pairs search (O(N^2)). The brute-force version
        becomes impractically slow once the sparse model has
        more than a few thousand points, which in practice
        forces overly aggressive point filtering just to keep
        initialization fast -- directly reducing the number of
        Gaussians the model starts training with. The KD-tree
        version handles tens/hundreds of thousands of points in
        seconds, so filtering thresholds can be set based on
        actual point quality instead of runtime constraints.
        """

        from scipy.spatial import cKDTree

        positions = np.asarray(
            [point["xyz"] for point in points],
            dtype=np.float32,
        )

        count = len(positions)

        if count < 2:
            raise RuntimeError(
                "At least two 3D points are required "
                "to estimate Gaussian scales."
            )

        print(
            "Estimating Gaussian scales (KD-tree):",
            count,
            "points",
        )

        tree = cKDTree(positions)

        distances, _ = tree.query(
            positions,
            k=4,
            workers=-1,
        )

        nearest_distance = np.mean(
            distances[:, 1:4],
            axis=1,
        ).astype(np.float32)

        nearest_distance = np.maximum(
            nearest_distance,
            1e-4,
        )

        median_distance = np.median(
            nearest_distance
        )

        lower_bound = max(
            median_distance * 0.1,
            1e-4,
        )

        upper_bound = max(
            median_distance * 10.0,
            lower_bound,
        )

        nearest_distance = np.clip(
            nearest_distance,
            lower_bound,
            upper_bound,
        )

        scale = np.maximum(
            nearest_distance * 0.5,
            1e-4,
        ).astype(np.float32)

        scales = np.repeat(
            scale[:, None],
            3,
            axis=1,
        )

        return scales

    def initialize(
        self,
        sparse_path,
        output_path,
    ):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        reconstruction = self._load_reconstruction(
            sparse_path
        )

        points = self._extract_points(
            reconstruction
        )

        points = self._filter_points(
            points
        )

        scales = self._estimate_scales(
            points
        )

        positions = np.asarray(
            [point["xyz"] for point in points],
            dtype=np.float32,
        )

        colors = np.asarray(
            [point["color"] for point in points],
            dtype=np.float32,
        )

        tracks = np.asarray(
            [point["track_length"] for point in points],
            dtype=np.int32,
        )

        errors = np.asarray(
            [point["error"] for point in points],
            dtype=np.float32,
        )

        np.savez_compressed(
            output_path,
            positions=positions,
            colors=colors,
            scales=scales,
            track_lengths=tracks,
            reprojection_errors=errors,
        )

        return {
            "output": str(output_path),
            "points": len(points),
            "mean_scale": float(
                scales.mean()
            ),
            "min_scale": float(
                scales.min()
            ),
            "max_scale": float(
                scales.max()
            ),
            "mean_track_length": float(
                tracks.mean()
            ),
            "mean_reprojection_error": float(
                errors.mean()
            ),
        }