import cv2
import numpy as np


class GlobalPairGenerator:
    def __init__(self, records, config):
        self.records = records
        matching = config["matching"]
        self.max_candidates = matching["global_candidates"]
        self.similarity_threshold = matching["global_similarity"]

    def _signature(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")

        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        signature = image.astype(np.float32)
        signature -= signature.mean()

        norm = np.linalg.norm(signature)
        if norm > 0:
            signature /= norm

        return signature

    @staticmethod
    def _image_path(record):
        for key in ("keyframe_path", "image_path", "selected_path", "path"):
            value = record.get(key)
            if value:
                return value
        raise KeyError(f"Record has no usable image path: {record}")

    def generate(self):
        signatures = []

        # IMPORTANT: append for every record. The previous version had
        # the append inside the wrong conditional/indentation level.
        for record in self.records:
            image_path = self._image_path(record)
            signatures.append(self._signature(image_path))

        pairs = set()

        for i, signature in enumerate(signatures):
            similarities = []

            for j, other in enumerate(signatures):
                if i == j:
                    continue

                similarity = float(np.sum(signature * other))
                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, j))

            similarities.sort(reverse=True)

            for _, j in similarities[: self.max_candidates]:
                a = self.records[i]["frame_id"]
                b = self.records[j]["frame_id"]
                if a != b:
                    pairs.add(tuple(sorted((a, b))))

        return sorted(pairs)
