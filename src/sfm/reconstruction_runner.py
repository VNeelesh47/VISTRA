from pathlib import Path
import json

import pycolmap

from src.sfm.database_validator import DatabaseValidator


class ReconstructionRunner:
    def __init__(self, database_path, image_path, output_path, config):
        self.database_path = Path(database_path)
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.config = config

    def _validate_database(self):
        validator = DatabaseValidator(
            self.database_path,
            self.config,
        )
        return validator.validate()

    def _create_options(self):
        # pycolmap 4.2.0:
        # - GlobalPipelineOptions owns multiple_models/min_model_size
        # - GlobalPipelineOptions.mapper owns mapper-specific options
        options = pycolmap.GlobalPipelineOptions()

        reconstruction_config = self.config.get(
            "reconstruction",
            {},
        )
        glomap_config = reconstruction_config.get(
            "glomap",
            {},
        )

        options.multiple_models =   False
        options.min_model_size = max(
            10,
            int(
                glomap_config.get(
                    "min_model_size",
                    10,
                )
            ),
        )

        mapper = options.mapper

        mapper.num_threads = glomap_config.get(
            "num_threads",
            -1,
        )
        mapper.random_seed = glomap_config.get(
            "random_seed",
            -1,
        )

        mapper.refine_sensor_from_rig = True

        # pycolmap 4.2.0 global positioning options.
        global_positioning = mapper.global_positioning
        global_positioning.use_gpu = True
        global_positioning.gpu_index = str(
        glomap_config.get(
            "gpu_index",
            0,
        )
    )

        # pycolmap 4.2.0 bundle adjustment options.
        ba = mapper.bundle_adjustment
        ba.refine_focal_length = True
        ba.refine_principal_point = False
        ba.refine_extra_params = True
        ba.refine_points3D = True

        # gs/BA CUDA support in pycolmap 4.2.0.
        ba.ceres.use_gpu = True
        ba.ceres.gpu_index = "0"

        return options

    def _validate_output(self):
        models = []

        if not self.output_path.exists():
            raise RuntimeError(
                "GLOMAP output directory does not exist."
            )

        for path in sorted(self.output_path.iterdir()):
            if not path.is_dir():
                continue

            cameras = path / "cameras.bin"
            images = path / "images.bin"
            points = path / "points3D.bin"

            if (
                cameras.exists()
                and images.exists()
                and points.exists()
            ):
                reconstruction = pycolmap.Reconstruction(
                    str(path)
                )

                models.append(
                    {
                        "path": str(path),
                        "cameras": reconstruction.num_cameras(),
                        "images": reconstruction.num_images(),
                        "points3D": reconstruction.num_points3D(),
                    }
                )

        if not models:
            raise RuntimeError(
                "GLOMAP completed without producing a valid sparse model."
            )

        expected_images = int(
            self.config["selection"].get(
                "min_frames",
                70,
            )
        )

        best = max(
            models,
            key=lambda model: (
                model["images"],
                model["points3D"],
            ),
        )

        total_images = len(
        list(self.image_path.glob("*"))
        )       

        registered_ratio = (
            best["images"]
            / max(total_images, 1)
        )

        print(
            f"Registered-image ratio: "
            f"{registered_ratio:.3f}"
        )

        print(
            f"Best reconstruction: "
            f"{best['images']} images, "
            f"{best['points3D']} points"
        )

        print(
            f"Registered-image ratio: "
            f"{registered_ratio:.3f}"
        )

        if best["images"] < 10:
            raise RuntimeError(
                "GLOMAP reconstruction is too small: "
                f"{best['images']} registered images."
            )
        if best["points3D"] < 300:
            raise RuntimeError(
                "GLOMAP reconstruction has too few 3D points: "
                f"{best['points3D']}."
            )
        models.sort(
                key=lambda model: (
                    model["images"],
                    model["points3D"],
                ),
                reverse=True,
            )

        return models

    def run(self):
        import shutil

        self.output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # REMOVE STALE RECONSTRUCTIONS
        # ---------------------------------------------------------

        for path in self.output_path.iterdir():
            if path.is_dir():
                shutil.rmtree(
                    path
                )
            elif path.is_file():
                path.unlink()

        # ---------------------------------------------------------
        # VALIDATE DATABASE
        # ---------------------------------------------------------

        database_stats = (
            self._validate_database()
        )

        options = self._create_options()

        print(
            "Running pycolmap 4.2.0 global mapping..."
        )

        pycolmap.global_mapping(
            database_path=str(
                self.database_path
            ),
            image_path=str(
                self.image_path
            ),
            output_path=str(
                self.output_path
            ),
            options=options,
        )

        models = self._validate_output()

        result = {
            "database": database_stats,
            "models": models,
            "selected_model": models[0],
        }

        report_path = (
            self.output_path
            / "reconstruction_report.json"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        return result
