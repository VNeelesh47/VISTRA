class PairGenerator:
    def __init__(self, records, config):
        self.records = records
        self.window = config["matching"]["temporal_window"]

    def generate(self):
        pairs = set()
        count = len(self.records)

        for i in range(count):
            for j in range(i + 1, min(i + self.window + 1, count)):
                pairs.add((self.records[i]["frame_id"], self.records[j]["frame_id"]))

        return sorted(pairs)