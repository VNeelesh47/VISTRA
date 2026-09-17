import av


class VideoDecoder:
    def __init__(self, video_path):
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.stream = self.container.streams.video[0]

    @property
    def fps(self):
        return float(self.stream.average_rate)

    @property
    def duration(self):
        if self.stream.duration is None:
            return float(self.container.duration / av.time_base)
        return float(self.stream.duration * self.stream.time_base)

    @property
    def width(self):
        return self.stream.width

    @property
    def height(self):
        return self.stream.height

    def frames(self):
        for frame in self.container.decode(self.stream):
            yield frame

    def close(self):
        self.container.close()