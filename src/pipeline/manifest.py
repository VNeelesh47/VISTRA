import json
from pathlib import Path


class PipelineManifest:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {
            "frames": [],
            "keyframes": [],
            "features": [],
            "matches": [],
            "reconstruction": {},
        }

    def add_frame(self, record):
        self.data["frames"].append(record)

    def add_keyframe(self, record):
        self.data["keyframes"].append(record)

    def add_feature(self, record):
        self.data["features"].append(record)

    def add_match(self, record):
        self.data["matches"].append(record)

    def set_reconstruction(self, record):
        self.data["reconstruction"] = record

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2)

    def load(self):
        with self.path.open("r", encoding="utf-8") as file:
            self.data = json.load(file)

        return self.data

    def get_keyframes(self):
        return self.data["keyframes"]

    def get_features(self):
        return self.data["features"]

    def get_matches(self):
        return self.data["matches"]