import pycolmap


class CameraBuilder:
    def __init__(self, config):
        camera_config = config["camera"]

        self.model = str(
            camera_config.get(
                "model",
                "SIMPLE_RADIAL",
            )
        ).upper()

        self.focal_factor = float(
            camera_config.get(
                "focal_length_factor",
                1.2,
            )
        )

        if self.focal_factor <= 0:
            raise ValueError(
                "camera.focal_length_factor must be > 0."
            )

        if self.focal_factor < 0.7 or self.focal_factor > 2.5:
            raise ValueError(
                "camera.focal_length_factor is outside "
                "the safe range [0.7, 2.5]."
            )

    def create(self, width, height):
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid camera resolution: {width} x {height}"
            )

        # COLMAP-style generic focal-length prior.
        #
        # For unknown drone-camera calibration, use a focal
        # prior based on the image dimensions rather than
        # assuming that the camera has a known calibration.
        #
        # GLOMAP later refines focal length and radial distortion.
        focal = self.focal_factor * max(
            width,
            height,
        )

        cx = width * 0.5
        cy = height * 0.5

        if self.model == "SIMPLE_RADIAL":

            params = [
                focal,
                cx,
                cy,
                0.0,
            ]

        elif self.model == "PINHOLE":

            params = [
                focal,
                focal,
                cx,
                cy,
            ]

        elif self.model == "SIMPLE_PINHOLE":

            params = [
                focal,
                cx,
                cy,
            ]

        else:
            raise ValueError(
                f"Unsupported camera model: {self.model}"
            )

        camera = pycolmap.Camera(
            model=self.model,
            width=width,
            height=height,
            params=params,
        )

        return camera