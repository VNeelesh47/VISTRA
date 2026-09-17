from pathlib import Path

import numpy as np
import pycolmap


class COLMAPDatabase:
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database = pycolmap.Database.open(str(self.database_path))

    def add_camera(self, camera_model, width, height, params):
        camera = pycolmap.Camera(
            model=camera_model,
            width=width,
            height=height,
            params=np.asarray(params, dtype=np.float64),
        )
        return self.database.write_camera(camera)

    def add_image(self, name, camera_id):
        image = pycolmap.Image(
            name=name,
            camera_id=camera_id,
        )
        return self.database.write_image(image)

    def add_keypoints(self, image_id, keypoints):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        self.database.write_keypoints(image_id, keypoints)

    def add_matches(self, image_id1, image_id2, matches):
        matches = np.asarray(matches, dtype=np.uint32)
        self.database.write_matches(image_id1, image_id2, matches)

    def add_geometry(self, image_id1, image_id2, geometry):
        self.database.write_two_view_geometry(
            image_id1,
            image_id2,
            geometry,
        )

    def close(self):
        self.database.close()