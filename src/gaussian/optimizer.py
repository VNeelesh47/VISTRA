from pathlib import Path

import numpy as np
import torch


class GaussianOptimizer:

    def __init__(
        self,
        config,
    ):

        self.config = config

        self.device = torch.device(
            "cuda"
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required for Gaussian optimization."
            )

        optimization = (
            config["gaussian"]["optimization"]
        )

        # =========================================================
        # LEARNING RATES
        # =========================================================

        # Position learning rate.
        #
        # Recommended for the current pipeline:
        #
        #     0.0002
        #
        # Trainer.py applies its own warmup + decay schedule
        # to this optimizer during training.

        self.lr_position = float(
            optimization.get(
                "learning_rate_position",
                0.0002,
            )
        )

        # Gaussian scale learning rate.

        self.lr_scale = float(
            optimization.get(
                "learning_rate_scale",
                0.005,
            )
        )

        # Gaussian rotation learning rate.

        self.lr_rotation = float(
            optimization.get(
                "learning_rate_rotation",
                0.001,
            )
        )

        # Gaussian opacity learning rate.

        self.lr_opacity = float(
            optimization.get(
                "learning_rate_opacity",
                0.05,
            )
        )

        # Spherical Harmonics learning rate.

        self.lr_sh = float(
            optimization.get(
                "learning_rate_sh",
                0.0025,
            )
        )

        # =========================================================
        # SH CONFIGURATION
        # =========================================================

        self.sh_degree = int(
            optimization.get(
                "sh_degree",
                3,
            )
        )

        if self.sh_degree < 0:

            raise ValueError(
                "sh_degree must be >= 0."
            )

        if self.sh_degree > 3:

            raise ValueError(
                "This optimizer supports SH degree up to 3."
            )

        # Number of real SH coefficients:
        #
        # degree 0 -> 1
        # degree 1 -> 4
        # degree 2 -> 9
        # degree 3 -> 16

        self.sh_coefficients = (
            (self.sh_degree + 1) ** 2
        )

        # =========================================================
        # INITIAL OPACITY
        # =========================================================

        self.initial_opacity = float(
            optimization.get(
                "initial_opacity",
                0.10,
            )
        )

        self.initial_opacity = max(
            1e-4,
            min(
                self.initial_opacity,
                1.0 - 1e-4,
            ),
        )

        # =========================================================
        # SCALE SAFETY
        # =========================================================

        self.minimum_scale = float(
            optimization.get(
                "minimum_scale",
                1e-4,
            )
        )

        self.minimum_scale = max(
            self.minimum_scale,
            1e-6,
        )

    # =============================================================
    # LOAD INITIALIZATION
    # =============================================================

    def _load_initialization(
        self,
        initialization_path,
    ):

        initialization_path = Path(
            initialization_path
        )

        if not initialization_path.exists():

            raise FileNotFoundError(
                "Gaussian initialization not found: "
                f"{initialization_path}"
            )

        data = np.load(
            initialization_path
        )

        # =========================================================
        # POSITIONS
        # =========================================================

        if "positions" not in data:

            raise KeyError(
                "Gaussian initialization does not contain "
                "'positions'."
            )

        positions_np = data[
            "positions"
        ]

        # =========================================================
        # COLORS
        # =========================================================

        if "colors" not in data:

            raise KeyError(
                "Gaussian initialization does not contain "
                "'colors'."
            )

        colors_np = data[
            "colors"
        ]

        # =========================================================
        # SCALES
        # =========================================================

        if "scales" not in data:

            raise KeyError(
                "Gaussian initialization does not contain "
                "'scales'."
            )

        scales_np = data[
            "scales"
        ]

        # =========================================================
        # CONVERT TO TORCH
        # =========================================================

        means = torch.from_numpy(
            positions_np
        ).float().to(
            self.device
        )

        colors = torch.from_numpy(
            colors_np
        ).float().to(
            self.device
        )

        scales = torch.from_numpy(
            scales_np
        ).float().to(
            self.device
        )

        # =========================================================
        # VALIDATE SHAPES
        # =========================================================

        if means.ndim != 2:

            raise ValueError(
                "positions must have shape [N, 3]."
            )

        if means.shape[1] != 3:

            raise ValueError(
                "positions must have shape [N, 3]."
            )

        if colors.ndim != 2:

            raise ValueError(
                "colors must have shape [N, 3]."
            )

        if colors.shape[1] != 3:

            raise ValueError(
                "colors must have shape [N, 3]."
            )

        if scales.ndim != 2:

            raise ValueError(
                "scales must have shape [N, 3]."
            )

        if scales.shape[1] != 3:

            raise ValueError(
                "scales must have shape [N, 3]."
            )

        if (
            means.shape[0]
            != colors.shape[0]
        ):

            raise ValueError(
                "positions and colors contain "
                "different numbers of points."
            )

        if (
            means.shape[0]
            != scales.shape[0]
        ):

            raise ValueError(
                "positions and scales contain "
                "different numbers of points."
            )

        # =========================================================
        # RGB NORMALIZATION
        # =========================================================
        #
        # Supports both:
        #
        #     uint8-style RGB: 0 ... 255
        #
        # and:
        #
        #     normalized RGB: 0 ... 1
        #
        # =========================================================

        if colors.numel() > 0:

            if float(
                colors.max()
            ) > 1.0:

                colors = (
                    colors / 255.0
                )

        colors = torch.nan_to_num(
            colors,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

        colors = torch.clamp(
            colors,
            min=1e-4,
            max=1.0 - 1e-4,
        )

        # =========================================================
        # SCALE SAFETY
        # =========================================================

        scales = torch.nan_to_num(
            scales,
            nan=self.minimum_scale,
            posinf=1.0,
            neginf=self.minimum_scale,
        )

        scales = torch.clamp(
            scales,
            min=self.minimum_scale,
        )

        # =========================================================
        # POSITION SAFETY
        # =========================================================

        means = torch.nan_to_num(
            means,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return (
            means,
            colors,
            scales,
        )

    # =============================================================
    # RGB -> SH
    # =============================================================

    def _rgb_to_sh(
        self,
        colors,
    ):

        # =========================================================
        # IMPORTANT
        # =========================================================
        #
        # SH coefficients are passed directly to gsplat.
        #
        # They must NOT be sigmoid logits.
        #
        # Degree-0 SH stores the initial RGB appearance.
        #
        # Higher-order coefficients start at zero.
        #
        # =========================================================

        count = colors.shape[0]

        shs = torch.zeros(
            (
                count,
                self.sh_coefficients,
                3,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # =========================================================
        # DEGREE-0 SH
        # =========================================================
        #
        # gsplat evaluates:
        #
        #     RGB = 0.5 + SH0 * C0
        #
        # where:
        #
        #     C0 = 0.28209479177387814
        #
        # Therefore:
        #
        #     SH0 = (RGB - 0.5) / C0
        #
        # =========================================================

        C0 = 0.28209479177387814

        shs[:, 0, :] = (
            colors - 0.5
        ) / C0

        # =========================================================
        # HIGHER-ORDER SH
        # =========================================================
        #
        # Start from zero.
        #
        # Trainer progressively enables:
        #
        #     degree 0
        #     degree 1
        #     degree 2
        #     degree 3
        #
        # =========================================================

        if self.sh_coefficients > 1:

            shs[
                :,
                1:,
                :
            ] = 0.0

        return shs

    # =============================================================
    # CREATE PARAMETERS
    # =============================================================

    def _create_parameters(
        self,
        means,
        colors,
        scales,
    ):

        count = means.shape[0]

        # =========================================================
        # ROTATIONS
        # =========================================================
        #
        # gsplat quaternion convention:
        #
        #     w x y z
        #
        # Identity:
        #
        #     [1, 0, 0, 0]
        #
        # =========================================================

        rotations = torch.zeros(
            (
                count,
                4,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        rotations[:, 0] = 1.0

        # =========================================================
        # OPACITY
        # =========================================================

        initial_opacity = torch.full(
            (
                count,
            ),
            self.initial_opacity,
            device=self.device,
            dtype=torch.float32,
        )

        opacity_logits = torch.logit(
            torch.clamp(
                initial_opacity,
                min=1e-4,
                max=1.0 - 1e-4,
            )
        )

        # =========================================================
        # LOG-SCALE REPRESENTATION
        # =========================================================
        #
        # Trainer/renderer uses:
        #
        #     real_scale = exp(parameters["scales"])
        #
        # Therefore we store:
        #
        #     log(real_scale)
        #
        # =========================================================

        scale_logits = torch.log(
            torch.clamp(
                scales,
                min=self.minimum_scale,
            )
        )

        # =========================================================
        # SH PARAMETERS
        # =========================================================

        shs = self._rgb_to_sh(
            colors
        )

        # =========================================================
        # TRAINABLE PARAMETERS
        # =========================================================

        parameters = {

            "means": torch.nn.Parameter(
                means
            ),

            "scales": torch.nn.Parameter(
                scale_logits
            ),

            "quats": torch.nn.Parameter(
                rotations
            ),

            "opacities": torch.nn.Parameter(
                opacity_logits
            ),

            "shs": torch.nn.Parameter(
                shs
            ),
        }

        return parameters

    # =============================================================
    # CREATE OPTIMIZERS
    # =============================================================

    def _create_optimizers(
        self,
        parameters,
    ):

        # =========================================================
        # SEPARATE OPTIMIZER FOR EACH PARAMETER TYPE
        # =========================================================
        #
        # This is required by trainer.py because:
        #
        #     means
        #
        # has its learning rate dynamically modified during
        # training.
        #
        # =========================================================

        optimizers = {

            # -----------------------------------------------------
            # POSITION
            # -----------------------------------------------------

            "means": torch.optim.Adam(
                [
                    parameters[
                        "means"
                    ]
                ],
                lr=self.lr_position,
                eps=1e-15,
            ),

            # -----------------------------------------------------
            # SCALE
            # -----------------------------------------------------

            "scales": torch.optim.Adam(
                [
                    parameters[
                        "scales"
                    ]
                ],
                lr=self.lr_scale,
                eps=1e-15,
            ),

            # -----------------------------------------------------
            # ROTATION
            # -----------------------------------------------------

            "quats": torch.optim.Adam(
                [
                    parameters[
                        "quats"
                    ]
                ],
                lr=self.lr_rotation,
                eps=1e-15,
            ),

            # -----------------------------------------------------
            # OPACITY
            # -----------------------------------------------------

            "opacities": torch.optim.Adam(
                [
                    parameters[
                        "opacities"
                    ]
                ],
                lr=self.lr_opacity,
                eps=1e-15,
            ),

            # -----------------------------------------------------
            # SPHERICAL HARMONICS
            # -----------------------------------------------------

            "shs": torch.optim.Adam(
                [
                    parameters[
                        "shs"
                    ]
                ],
                lr=self.lr_sh,
                eps=1e-15,
            ),
        }

        return optimizers

    # =============================================================
    # INITIALIZE
    # =============================================================

    def initialize(
        self,
        initialization_path,
    ):

        # =========================================================
        # LOAD INITIAL POINT CLOUD
        # =========================================================

        (
            means,
            colors,
            scales,
        ) = self._load_initialization(
            initialization_path
        )

        # =========================================================
        # CREATE TRAINABLE PARAMETERS
        # =========================================================

        parameters = (
            self._create_parameters(
                means,
                colors,
                scales,
            )
        )

        # =========================================================
        # CREATE OPTIMIZERS
        # =========================================================

        optimizers = (
            self._create_optimizers(
                parameters
            )
        )

        # =========================================================
        # VALIDATE GAUSSIAN COUNT
        # =========================================================

        expected_count = (
            means.shape[0]
        )

        actual_count = (
            parameters[
                "means"
            ].shape[0]
        )

        if (
            expected_count
            != actual_count
        ):

            raise RuntimeError(
                "Gaussian count changed unexpectedly "
                "during initialization."
            )

        # =========================================================
        # VALIDATE SHAPE
        # =========================================================

        expected_sh_shape = (
            expected_count,
            self.sh_coefficients,
            3,
        )

        actual_sh_shape = (
            parameters[
                "shs"
            ].shape
        )

        if (
            tuple(actual_sh_shape)
            != tuple(expected_sh_shape)
        ):

            raise RuntimeError(
                "Invalid SH parameter shape. "
                f"Expected {expected_sh_shape}, "
                f"got {tuple(actual_sh_shape)}."
            )

        # =========================================================
        # VALIDATE FINITE VALUES
        # =========================================================

        for key, parameter in (
            parameters.items()
        ):

            if not torch.isfinite(
                parameter
            ).all():

                raise RuntimeError(
                    f"Gaussian parameter '{key}' "
                    "contains NaN or Inf values."
                )

        # =========================================================
        # REPORT
        # =========================================================

        print()

        print(
            "=" * 70
        )

        print(
            "GAUSSIAN PARAMETER INITIALIZATION"
        )

        print(
            "=" * 70
        )

        print(
            f"Gaussians:       {actual_count}"
        )

        print(
            f"SH degree:       {self.sh_degree}"
        )

        print(
            f"SH coefficients: {self.sh_coefficients}"
        )

        print(
            f"SH tensor:       {tuple(actual_sh_shape)}"
        )

        print(
            f"Position LR:     {self.lr_position:.6f}"
        )

        print(
            f"Scale LR:        {self.lr_scale:.6f}"
        )

        print(
            f"Rotation LR:     {self.lr_rotation:.6f}"
        )

        print(
            f"Opacity LR:      {self.lr_opacity:.6f}"
        )

        print(
            f"SH LR:           {self.lr_sh:.6f}"
        )

        print(
            f"Initial opacity: {self.initial_opacity:.4f}"
        )

        print(
            "=" * 70
        )

        print()

        return (
            parameters,
            optimizers,
        )

    # =============================================================
    # SAVE
    # =============================================================

    def save(
        self,
        parameters,
        output_path,
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =========================================================
        # SAVE ALL PARAMETERS
        # =========================================================

        save_data = {}

        for key, value in (
            parameters.items()
        ):

            save_data[
                key
            ] = value.detach().cpu()

        torch.save(
            save_data,
            output_path,
        )

        return output_path