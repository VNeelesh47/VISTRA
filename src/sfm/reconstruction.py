from pathlib import Path
import shutil

import pycolmap

from src.sfm.camera import CameraBuilder
from src.sfm.database_builder import DatabaseBuilder
from src.sfm.global_mapper import GlobalMapper


class SparseReconstruction:
    def __init__(self, records, feature_dir, reconstruction_dir, config):
        self.records = records
        self.feature_dir = Path(feature_dir)
        self.reconstruction_dir = Path(reconstruction_dir)
        self.config = config

    def run(self):
        self.reconstruction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        first = self.records[0]

        width = first["width"]
        height = first["height"]

        camera = CameraBuilder(self.config).create(
            width,
            height,
        )

        database_path = (
            self.reconstruction_dir / "database.db"
        )

        database_builder = DatabaseBuilder(
            database_path,
            self.records,
        )

        return camera, database_path