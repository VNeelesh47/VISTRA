import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.gaussian.dataset import GaussianDataset
from src.gaussian.renderer import GaussianRenderer


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(checkpoint_path, device):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Gaussian checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print("=" * 70)
    print("LOADING TRAINED GAUSSIAN CHECKPOINT")
    print("=" * 70)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "checkpoint must be a dictionary."
        )

    if "parameters" not in checkpoint:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "'parameters' is missing."
        )

    checkpoint_parameters = checkpoint["parameters"]

    if not isinstance(checkpoint_parameters, dict):
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "'parameters' must be a dictionary."
        )

    # --------------------------------------------------------
    # Required Gaussian geometry parameters
    # --------------------------------------------------------

    required = [
        "means",
        "scales",
        "quats",
        "opacities",
    ]

    parameters = {}

    for name in required:

        if name not in checkpoint_parameters:
            raise RuntimeError(
                "Invalid Gaussian checkpoint: "
                f"parameter '{name}' is missing."
            )

        tensor = checkpoint_parameters[name]

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        parameters[name] = torch.nn.Parameter(
            tensor.to(
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

    # --------------------------------------------------------
    # Appearance parameter
    #
    # Current GaussianRenderer supports either:
    #
    #     colors
    #
    # or:
    #
    #     shs
    #
    # Do NOT require colors unconditionally.
    # --------------------------------------------------------

    has_colors = "colors" in checkpoint_parameters
    has_shs = "shs" in checkpoint_parameters

    if not has_colors and not has_shs:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "neither 'colors' nor 'shs' is present."
        )

    if has_colors and has_shs:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "both 'colors' and 'shs' are present. "
            "The checkpoint must contain only one "
            "appearance representation."
        )

    if has_colors:

        tensor = checkpoint_parameters["colors"]

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        parameters["colors"] = torch.nn.Parameter(
            tensor.to(
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        appearance_name = "colors"

    else:

        tensor = checkpoint_parameters["shs"]

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        parameters["shs"] = torch.nn.Parameter(
            tensor.to(
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        appearance_name = "shs"

    # --------------------------------------------------------
    # Check Gaussian count consistency
    # --------------------------------------------------------

    gaussian_count = parameters["means"].shape[0]

    if parameters["scales"].shape[0] != gaussian_count:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "means and scales contain different "
            "numbers of Gaussians."
        )

    if parameters["quats"].shape[0] != gaussian_count:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "means and quats contain different "
            "numbers of Gaussians."
        )

    if parameters["opacities"].shape[0] != gaussian_count:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            "means and opacities contain different "
            "numbers of Gaussians."
        )

    if parameters[appearance_name].shape[0] != gaussian_count:
        raise RuntimeError(
            "Invalid Gaussian checkpoint: "
            f"means and {appearance_name} contain "
            "different numbers of Gaussians."
        )

    # --------------------------------------------------------
    # Checkpoint information
    # --------------------------------------------------------

    print(
        f"Training iteration: "
        f"{checkpoint.get('iteration', 'unknown')}"
    )

    print(
        f"Training loss: "
        f"{checkpoint.get('loss', 'unknown')}"
    )

    print(
        f"Gaussians: "
        f"{gaussian_count}"
    )

    print(
        f"Appearance: "
        f"{appearance_name}"
    )

    print(
        f"Means shape: "
        f"{tuple(parameters['means'].shape)}"
    )

    print(
        f"Scales shape: "
        f"{tuple(parameters['scales'].shape)}"
    )

    print(
        f"Quats shape: "
        f"{tuple(parameters['quats'].shape)}"
    )

    print(
        f"Opacities shape: "
        f"{tuple(parameters['opacities'].shape)}"
    )

    print(
        f"{appearance_name.capitalize()} shape: "
        f"{tuple(parameters[appearance_name].shape)}"
    )

    print()

    return parameters


# ============================================================
# TENSOR -> IMAGE
# ============================================================

def tensor_to_image(tensor):

    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
    )

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    image = (
        tensor.numpy() * 255.0
    ).astype(
        np.uint8
    )

    return image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(output_path),
        bgr,
    )

    if not success:
        raise RuntimeError(
            "Failed to save rendered image:\n"
            f"{output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Render a trained gsplat Gaussian checkpoint "
            "using COLMAP camera poses."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline YAML configuration.",
    )

    parser.add_argument(
        "--sparse",
        required=True,
        help="Path to COLMAP sparse reconstruction.",
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Path to training images.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained Gaussian checkpoint.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where rendered images are saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of frames to render. "
            "0 means all frames."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("STAGE 10 — TRAINED GAUSSIAN RENDERING")
    print("=" * 70)
    print()

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Gaussian rendering."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Torch:",
        torch.__version__,
    )

    print()

    # ========================================================
    # CONFIG
    # ========================================================

    config_path = Path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Configuration file not found:\n"
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    parameters = load_checkpoint(
        args.checkpoint,
        device,
    )

    # ========================================================
    # LOAD COLMAP DATASET
    # ========================================================

    print("=" * 70)
    print("LOADING COLMAP CAMERAS")
    print("=" * 70)

    dataset = GaussianDataset(
        args.sparse,
        args.images,
        config,
        device="cuda",
    )

    print(
        f"Posed frames: {len(dataset)}"
    )

    print()

    # ========================================================
    # RENDERER
    # ========================================================

    renderer = GaussianRenderer(
        config
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FRAME COUNT
    # ========================================================

    total_frames = len(
        dataset
    )

    if args.limit > 0:

        frame_count = min(
            args.limit,
            total_frames,
        )

    else:

        frame_count = total_frames

    print("=" * 70)
    print("RENDERING")
    print("=" * 70)

    print(
        f"Frames to render: {frame_count}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print()

    # ========================================================
    # RENDER
    #
    # IMPORTANT:
    #
    # DO NOT use torch.no_grad() here.
    #
    # gsplat must create differentiable means2d metadata.
    #
    # We do not call backward(), so no training occurs.
    # ========================================================

    for index in tqdm(
        range(frame_count),
        desc="Rendering",
        unit="frame",
    ):

        frame = dataset[
            index
        ]

        result = renderer.render(
            parameters,
            frame,
        )

        # ----------------------------------------------------
        # Validate renderer output
        # ----------------------------------------------------

        if "image" not in result:
            raise RuntimeError(
                "Gaussian renderer did not return "
                "'image'."
            )

        if "means2d" not in result:
            raise RuntimeError(
                "Gaussian renderer did not return "
                "'means2d'."
            )

        means2d = result[
            "means2d"
        ]

        if means2d is None:
            raise RuntimeError(
                "Gaussian renderer returned "
                "'means2d' as None."
            )

        if not means2d.requires_grad:
            raise RuntimeError(
                "Gaussian renderer returned "
                "'means2d' without gradients."
            )

        # ----------------------------------------------------
        # Convert rendered image
        # ----------------------------------------------------

        rendered = result[
            "image"
        ]

        image = tensor_to_image(
            rendered
        )

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        original_name = Path(
            frame["name"]
        )

        output_name = (
            f"{index:05d}_"
            f"{original_name.stem}.png"
        )

        output_path = (
            output_dir
            / output_name
        )

        save_image(
            image,
            output_path,
        )

    # ========================================================
    # CUDA SYNC
    # ========================================================

    torch.cuda.synchronize()

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 70)
    print("STAGE 10 COMPLETE")
    print("=" * 70)

    print(
        f"Rendered frames: {frame_count}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()