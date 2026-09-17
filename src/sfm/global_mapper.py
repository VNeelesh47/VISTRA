from pathlib import Path

import pycolmap


class GlobalMapper:
    def __init__(self, config):
        self.config = config

    def run(self, database_path, image_path, output_path):
        database_path = Path(database_path)
        image_path = Path(image_path)
        output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        options = pycolmap.GlobalPipelineOptions()

        options.global_mapper.multiple_models = True
        options.global_mapper.min_model_size = 3

        options.global_mapper.gp_use_gpu = True
        options.global_mapper.gp_gpu_index = -1

        options.global_mapper.ba_ceres_use_gpu = True
        options.global_mapper.ba_gpu_index = -1

        options.global_mapper.ba_refine_focal_length = True
        options.global_mapper.ba_refine_principal_point = False
        options.global_mapper.ba_refine_extra_params = True
        options.global_mapper.ba_refine_points3D = True

        return pycolmap.global_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(output_path),
            options=options,
        )