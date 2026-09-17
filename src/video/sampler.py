import json
from pathlib import Path

import cv2
from tqdm import tqdm

from src.video.decoder import VideoDecoder


class VideoSampler:
    def __init__(self, video_path, output_dir, config):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.sampling = config["sampling"]

    def run(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        decoder = VideoDecoder(self.video_path)

        source_fps = decoder.fps
        target_fps = min(
            self.sampling["target_fps"],
            source_fps,
        )

        interval = source_fps / target_fps

        records = []
        next_frame = 0.0
        frame_index = 0
        saved_index = 0

        # Estimate total frames for progress reporting.
        total_frames = None

        if source_fps > 0 and decoder.duration:
            total_frames = int(
                decoder.duration * source_fps
            )

        try:
            with tqdm(
                total=total_frames,
                desc="Sampling video",
                unit="frame",
            ) as progress:

                for frame in decoder.frames():

                    if frame_index >= next_frame:
                        image = frame.to_ndarray(
                            format="bgr24"
                        )

                        filename = (
                            f"{saved_index:06d}."
                            f"{self.sampling['image_format']}"
                        )

                        output_path = (
                            self.output_dir / filename
                        )

                        if (
                            self.sampling[
                                "image_format"
                            ].lower()
                            in {"jpg", "jpeg"}
                        ):
                            cv2.imwrite(
                                str(output_path),
                                image,
                                [
                                    cv2.IMWRITE_JPEG_QUALITY,
                                    self.sampling[
                                        "jpeg_quality"
                                    ],
                                ],
                            )
                        else:
                            cv2.imwrite(
                                str(output_path),
                                image,
                            )

                        records.append(
                            {
                                "frame_id": saved_index,
                                "source_index": frame_index,
                                "timestamp": float(
                                    frame.time or 0.0
                                ),
                                "path": str(output_path),
                                "keyframe_path": str(
                                    output_path
                                ),
                                "width": image.shape[1],
                                "height": image.shape[0],
                            }
                        )

                        saved_index += 1
                        next_frame += interval

                    frame_index += 1
                    progress.update(1)

        finally:
            decoder.close()

        metadata_path = (
            self.output_dir / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "video": str(
                        self.video_path
                    ),
                    "source_fps": source_fps,
                    "target_fps": target_fps,
                    "duration": decoder.duration,
                    "width": decoder.width,
                    "height": decoder.height,
                    "frames": records,
                },
                file,
                indent=2,
            )

        return records

