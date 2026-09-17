from pathlib import Path
import argparse
import struct

import numpy as np
import torch


# =============================================================
# LOAD CHECKPOINT
# =============================================================

def load_checkpoint(checkpoint_path):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        "=" * 70
    )

    print(
        "LOADING GAUSSIAN CHECKPOINT"
    )

    print(
        "=" * 70
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # ---------------------------------------------------------
    # Support:
    #
    # 1. Trainer checkpoint:
    #
    #    {
    #        "iteration": ...,
    #        "parameters": {...},
    #        "loss": ...,
    #        "gaussians": ...
    #    }
    #
    # 2. Direct parameter checkpoint:
    #
    #    {
    #        "means": ...,
    #        ...
    #    }
    # ---------------------------------------------------------

    if "parameters" in checkpoint:

        parameters = checkpoint[
            "parameters"
        ]

    else:

        parameters = checkpoint

    required = [
        "means",
        "scales",
        "quats",
        "opacities",
    ]

    for key in required:

        if key not in parameters:

            raise KeyError(
                "Checkpoint is missing Gaussian "
                f"parameter: {key}"
            )

    # ---------------------------------------------------------
    # Appearance
    #
    # New pipeline:
    #
    #     shs
    #
    # Legacy pipeline:
    #
    #     colors
    # ---------------------------------------------------------

    if "shs" not in parameters:

        if "colors" not in parameters:

            raise KeyError(
                "Checkpoint contains neither "
                "'shs' nor 'colors'."
            )

    means = (
        parameters["means"]
        .detach()
        .cpu()
        .numpy()
    )

    scales = (
        parameters["scales"]
        .detach()
        .cpu()
        .numpy()
    )

    quats = (
        parameters["quats"]
        .detach()
        .cpu()
        .numpy()
    )

    opacities = (
        parameters["opacities"]
        .detach()
        .cpu()
        .numpy()
    )

    shs = None

    colors = None

    if "shs" in parameters:

        shs = (
            parameters["shs"]
            .detach()
            .cpu()
            .numpy()
        )

    elif "colors" in parameters:

        colors = (
            parameters["colors"]
            .detach()
            .cpu()
            .numpy()
        )

    iteration = checkpoint.get(
        "iteration",
        None
    )

    loss = checkpoint.get(
        "loss",
        None
    )

    print()

    print(
        f"Gaussians: {len(means)}"
    )

    if iteration is not None:

        print(
            f"Iteration: {iteration}"
        )

    if loss is not None:

        print(
            f"Loss:      {float(loss):.6f}"
        )

    if shs is not None:

        print(
            f"SH shape:  {shs.shape}"
        )

    elif colors is not None:

        print(
            f"Color shape: {colors.shape}"
        )

    return (
        means,
        scales,
        quats,
        opacities,
        shs,
        colors,
    )


# =============================================================
# SIGMOID
# =============================================================

def sigmoid(x):

    x = np.clip(
        x,
        -80.0,
        80.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


# =============================================================
# QUATERNION NORMALIZATION
# =============================================================

def quaternion_normalize(
    quats,
):

    norms = np.linalg.norm(
        quats,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-8,
    )

    return (
        quats
        / norms
    )


# =============================================================
# SH -> RGB
# =============================================================

def sh_to_rgb(
    shs,
):
    """
    Convert degree-0 SH coefficients into RGB.

    The optimizer initializes:

        sh0 = (RGB - 0.5) / C0

    where:

        C0 = 0.28209479177387814

    Therefore:

        RGB = sh0 * C0 + 0.5

    Only the degree-0 coefficient is required for
    a view-independent RGB representation.

    Higher-order SH coefficients remain available
    separately in the checkpoint but are not represented
    by ordinary RGB PLY fields.
    """

    if shs.ndim != 3:

        raise ValueError(
            "SH tensor must have shape "
            "[N, K, 3]. "
            f"Got {shs.shape}"
        )

    if shs.shape[-1] != 3:

        raise ValueError(
            "SH tensor last dimension must "
            f"be 3. Got {shs.shape}"
        )

    if shs.shape[1] < 1:

        raise ValueError(
            "SH tensor contains no degree-0 "
            "coefficient."
        )

    C0 = 0.28209479177387814

    rgb = (
        shs[:, 0, :]
        * C0
        + 0.5
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    return rgb


# =============================================================
# PREPARE GAUSSIANS
# =============================================================

def prepare_gaussians(
    means,
    scales,
    quats,
    opacities,
    shs=None,
    colors=None,
):

    print()

    print(
        "=" * 70
    )

    print(
        "PREPARING GAUSSIANS"
    )

    print(
        "=" * 70
    )

    count = means.shape[0]

    print(
        f"Gaussian count: {count}"
    )

    # ---------------------------------------------------------
    # Validate means
    # ---------------------------------------------------------

    if (
        means.ndim != 2
        or means.shape[1] != 3
    ):

        raise ValueError(
            f"Invalid means shape: "
            f"{means.shape}"
        )

    # ---------------------------------------------------------
    # Validate scales
    # ---------------------------------------------------------

    if (
        scales.ndim != 2
        or scales.shape[1] != 3
    ):

        raise ValueError(
            f"Invalid scales shape: "
            f"{scales.shape}"
        )

    # ---------------------------------------------------------
    # Validate quaternions
    # ---------------------------------------------------------

    if (
        quats.ndim != 2
        or quats.shape[1] != 4
    ):

        raise ValueError(
            f"Invalid quaternion shape: "
            f"{quats.shape}"
        )

    # ---------------------------------------------------------
    # Validate opacity
    # ---------------------------------------------------------

    opacities = opacities.reshape(-1)

    if len(opacities) != count:

        raise ValueError(
            "Opacity count does not match "
            "Gaussian count."
        )

    # ---------------------------------------------------------
    # Convert log-scales to real scales
    #
    # Optimizer stores:
    #
    #     log(scale)
    #
    # Renderer uses:
    #
    #     exp(scale)
    # ---------------------------------------------------------

    scales = np.exp(
        np.clip(
            scales,
            -20.0,
            20.0,
        )
    )

    # ---------------------------------------------------------
    # Convert opacity logits
    #
    # Optimizer stores:
    #
    #     logit(opacity)
    #
    # Renderer uses:
    #
    #     sigmoid(opacity)
    # ---------------------------------------------------------

    opacities = sigmoid(
        opacities
    )

    # ---------------------------------------------------------
    # Appearance
    # ---------------------------------------------------------

    if shs is not None:

        rgb = sh_to_rgb(
            shs
        )

        appearance_mode = (
            "SH"
        )

    elif colors is not None:

        if (
            colors.ndim != 2
            or colors.shape[1] != 3
        ):

            raise ValueError(
                f"Invalid color shape: "
                f"{colors.shape}"
            )

        # -----------------------------------------------------
        # Legacy colors are stored as logits.
        # -----------------------------------------------------

        rgb = sigmoid(
            colors
        )

        appearance_mode = (
            "RGB"
        )

    else:

        raise RuntimeError(
            "No Gaussian appearance data."
        )

    colors_uint8 = np.clip(
        rgb * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    # ---------------------------------------------------------
    # Normalize quaternions
    # ---------------------------------------------------------

    quats = quaternion_normalize(
        quats
    )

    # ---------------------------------------------------------
    # Finite-value filtering
    # ---------------------------------------------------------

    valid = np.isfinite(
        means
    ).all(
        axis=1
    )

    valid &= np.isfinite(
        scales
    ).all(
        axis=1
    )

    valid &= np.isfinite(
        quats
    ).all(
        axis=1
    )

    valid &= np.isfinite(
        opacities
    )

    valid &= np.isfinite(
        rgb
    ).all(
        axis=1
    )

    means = means[
        valid
    ]

    scales = scales[
        valid
    ]

    quats = quats[
        valid
    ]

    opacities = opacities[
        valid
    ]

    colors_uint8 = colors_uint8[
        valid
    ]

    if shs is not None:

        shs = shs[
            valid
        ]

    print(
        f"Valid Gaussians: "
        f"{len(means)}"
    )

    print(
        f"Appearance:      "
        f"{appearance_mode}"
    )

    if len(means) == 0:

        raise RuntimeError(
            "No valid Gaussians remain "
            "after filtering."
        )

    return (
        means,
        scales,
        quats,
        opacities,
        colors_uint8,
        shs,
    )


# =============================================================
# WRITE GAUSSIAN PLY
# =============================================================

def write_ply(
    output_path,
    means,
    scales,
    quats,
    opacities,
    colors,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = len(
        means
    )

    print()

    print(
        "=" * 70
    )

    print(
        "WRITING GAUSSIAN PLY"
    )

    print(
        "=" * 70
    )

    print(
        f"Output:   {output_path}"
    )

    print(
        f"Vertices: {count}"
    )

    # ---------------------------------------------------------
    # Custom Gaussian PLY
    #
    # Fields:
    #
    # position
    # normal
    # RGB
    # opacity
    # scale
    # quaternion
    # ---------------------------------------------------------

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float opacity\n"
        "property float scale_x\n"
        "property float scale_y\n"
        "property float scale_z\n"
        "property float rot_w\n"
        "property float rot_x\n"
        "property float rot_y\n"
        "property float rot_z\n"
        "end_header\n"
    ).encode(
        "ascii"
    )

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            header
        )

        for index in range(
            count
        ):

            x, y, z = (
                means[index]
            )

            r = int(
                colors[
                    index,
                    0
                ]
            )

            g = int(
                colors[
                    index,
                    1
                ]
            )

            b = int(
                colors[
                    index,
                    2
                ]
            )

            sx, sy, sz = (
                scales[index]
            )

            rw, rx, ry, rz = (
                quats[index]
            )

            opacity = float(
                opacities[index]
            )

            # -------------------------------------------------
            # No traditional surface normal is stored by the
            # Gaussian representation.
            # -------------------------------------------------

            nx = 0.0
            ny = 0.0
            nz = 0.0

            file.write(
                struct.pack(
                    "<fff"
                    "fff"
                    "BBB"
                    "f"
                    "fff"
                    "ffff",

                    float(x),
                    float(y),
                    float(z),

                    float(nx),
                    float(ny),
                    float(nz),

                    r,
                    g,
                    b,

                    opacity,

                    float(sx),
                    float(sy),
                    float(sz),

                    float(rw),
                    float(rx),
                    float(ry),
                    float(rz),
                )
            )

    file_size_mb = (
        output_path.stat().st_size
        / (
            1024.0
            * 1024.0
        )
    )

    print()

    print(
        "PLY written successfully."
    )

    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )


# =============================================================
# WRITE SIMPLE POINT CLOUD
# =============================================================

def write_simple_ply(
    output_path,
    means,
    colors,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = len(
        means
    )

    print()

    print(
        "=" * 70
    )

    print(
        "WRITING SIMPLE POINT CLOUD"
    )

    print(
        "=" * 70
    )

    print(
        f"Output:   {output_path}"
    )

    print(
        f"Vertices: {count}"
    )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode(
        "ascii"
    )

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            header
        )

        for index in range(
            count
        ):

            x, y, z = (
                means[index]
            )

            r = int(
                colors[
                    index,
                    0
                ]
            )

            g = int(
                colors[
                    index,
                    1
                ]
            )

            b = int(
                colors[
                    index,
                    2
                ]
            )

            file.write(
                struct.pack(
                    "<fffBBB",

                    float(x),
                    float(y),
                    float(z),

                    r,
                    g,
                    b,
                )
            )

    file_size_mb = (
        output_path.stat().st_size
        / (
            1024.0
            * 1024.0
        )
    )

    print(
        f"Simple PLY written: "
        f"{output_path}"
    )

    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )


# =============================================================
# SAVE SH PARAMETERS
# =============================================================

def save_sh_parameters(
    output_path,
    shs,
):

    if shs is None:

        return

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        shs.astype(
            np.float32
        )
    )

    print(
        f"SH parameters saved: "
        f"{output_path}"
    )


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Export trained Gaussian "
            "checkpoint to PLY."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to trained Gaussian "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output Gaussian PLY path."
        ),
    )

    parser.add_argument(
        "--simple-output",
        default=None,
        help=(
            "Optional simple RGB point-cloud "
            "PLY for maximum viewer compatibility."
        ),
    )

    parser.add_argument(
        "--sh-output",
        default=None,
        help=(
            "Optional .npy output containing "
            "the trained SH coefficients."
        ),
    )

    args = parser.parse_args()

    print()

    print(
        "=" * 70
    )

    print(
        "SIH26158 — GAUSSIAN PLY EXPORT"
    )

    print(
        "=" * 70
    )

    (
        means,
        scales,
        quats,
        opacities,
        shs,
        colors,
    ) = load_checkpoint(
        args.checkpoint
    )

    (
        means,
        scales,
        quats,
        opacities,
        colors,
        shs,
    ) = prepare_gaussians(
        means,
        scales,
        quats,
        opacities,
        shs=shs,
        colors=colors,
    )

    # ---------------------------------------------------------
    # Main Gaussian PLY
    # ---------------------------------------------------------

    write_ply(
        args.output,
        means,
        scales,
        quats,
        opacities,
        colors,
    )

    # ---------------------------------------------------------
    # Simple point-cloud PLY
    # ---------------------------------------------------------

    if args.simple_output:

        write_simple_ply(
            args.simple_output,
            means,
            colors,
        )

    # ---------------------------------------------------------
    # Preserve SH coefficients
    # ---------------------------------------------------------

    if args.sh_output:

        save_sh_parameters(
            args.sh_output,
            shs,
        )

    print()

    print(
        "=" * 70
    )

    print(
        "EXPORT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Gaussian PLY: "
        f"{args.output}"
    )

    if args.simple_output:

        print(
            f"Simple PLY:   "
            f"{args.simple_output}"
        )

    if args.sh_output:

        print(
            f"SH data:      "
            f"{args.sh_output}"
        )

    print(
        "=" * 70
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()