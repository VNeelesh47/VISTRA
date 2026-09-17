from pathlib import Path


class PairSelector:
    def __init__(self, config):
        self.config = config

        matching = config["matching"]

        self.window_size = matching[
            "temporal_window"
        ]

        self.stride = matching[
            "pair_stride"
        ]

    def select(self, image_paths):
        image_paths = [
            Path(path)
            for path in image_paths
        ]

        image_paths.sort()

        pairs = []
        seen = set()

        count = len(image_paths)

        for i in range(
            0,
            count,
            self.stride,
        ):
            end = min(
                i + self.window_size + 1,
                count,
            )

            for j in range(
                i + 1,
                end,
            ):
                pair = (
                    str(image_paths[i]),
                    str(image_paths[j]),
                )

                if pair in seen:
                    continue

                seen.add(pair)
                pairs.append(pair)

        return pairs