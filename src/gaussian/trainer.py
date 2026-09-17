from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from gsplat import DefaultStrategy

from src.gaussian.renderer import GaussianRenderer


class GaussianTrainer:

    def __init__(
        self,
        dataset,
        parameters,
        optimizers,
        config,
    ):

        self.dataset = dataset
        self.parameters = parameters
        self.optimizers = optimizers
        self.config = config

        # =========================================================
        # DEVICE
        # =========================================================

        self.device = torch.device(
            "cuda"
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required for Gaussian training."
            )

        # =========================================================
        # OPTIMIZATION CONFIGURATION
        # =========================================================
        #
        # Your YAML places "optimization" at the TOP LEVEL:
        #
        # optimization:
        #     iterations: 30000
        #
        # However, older versions of the code expected:
        #
        # gaussian:
        #     optimization:
        #
        # Support both layouts so the trainer does not break.
        #
        # =========================================================

        optimization = config.get(
            "optimization",
            config.get(
                "gaussian",
                {}
            ).get(
                "optimization",
                {}
            ),
        )

        self.optimization = optimization

        # =========================================================
        # BASIC TRAINING SETTINGS
        # =========================================================

        self.iterations = int(
            optimization.get(
                "iterations",
                30000,
            )
        )

        self.checkpoint_interval = int(
            optimization.get(
                "checkpoint_interval",
                500,
            )
        )

        if self.iterations <= 0:

            raise ValueError(
                "iterations must be greater than zero."
            )

        if self.checkpoint_interval <= 0:

            raise ValueError(
                "checkpoint_interval must be greater than zero."
            )

        # =========================================================
        # LOSS WEIGHTS
        # =========================================================

        self.lambda_l1 = float(
            optimization.get(
                "lambda_l1",
                0.80,
            )
        )

        self.lambda_ssim = float(
            optimization.get(
                "lambda_ssim",
                0.20,
            )
        )

        self.lambda_edge = float(
            optimization.get(
                "lambda_edge",
                0.0,
            )
        )

        self.lambda_multiscale = float(
            optimization.get(
                "lambda_multiscale",
                0.0,
            )
        )

        if self.lambda_l1 < 0:
            raise ValueError(
                "lambda_l1 must be >= 0."
            )

        if self.lambda_ssim < 0:
            raise ValueError(
                "lambda_ssim must be >= 0."
            )

        if self.lambda_edge < 0:
            raise ValueError(
                "lambda_edge must be >= 0."
            )

        if self.lambda_multiscale < 0:
            raise ValueError(
                "lambda_multiscale must be >= 0."
            )

        if (
            self.lambda_l1
            + self.lambda_ssim
            + self.lambda_edge
            + self.lambda_multiscale
            <= 0
        ):

            raise ValueError(
                "At least one loss weight must be greater than zero."
            )

        # =========================================================
        # POSITION LEARNING RATE
        # =========================================================

        self.base_position_lr = float(
            optimization.get(
                "learning_rate_position",
                0.00016,
            )
        )

        self.position_lr_final_factor = float(
            optimization.get(
                "position_lr_final_factor",
                0.05,
            )
        )

        self.position_warmup = int(
            optimization.get(
                "position_warmup",
                500,
            )
        )

        if self.base_position_lr <= 0:

            raise ValueError(
                "learning_rate_position must be greater than zero."
            )

        self.position_lr_final_factor = max(
            self.position_lr_final_factor,
            1e-5,
        )

        self.position_warmup = max(
            self.position_warmup,
            0,
        )

        # =========================================================
        # ADAPTIVE FRAME SAMPLING
        # =========================================================
        #
        # These settings are optional.
        #
        # If not present in YAML, normal random frame sampling
        # is used.
        #
        # =========================================================

        self.adaptive_sampling = bool(
            optimization.get(
                "adaptive_frame_sampling",
                False,
            )
        )

        self.hard_frame_probability = float(
            optimization.get(
                "hard_frame_probability",
                0.35,
            )
        )

        self.uniform_frame_probability = float(
            optimization.get(
                "uniform_frame_probability",
                0.65,
            )
        )

        self.frame_loss_momentum = float(
            optimization.get(
                "frame_loss_momentum",
                0.90,
            )
        )

        self.hard_frame_probability = max(
            0.0,
            min(
                1.0,
                self.hard_frame_probability,
            ),
        )

        self.uniform_frame_probability = max(
            0.0,
            min(
                1.0,
                self.uniform_frame_probability,
            ),
        )

        self.frame_loss_momentum = max(
            0.0,
            min(
                0.9999,
                self.frame_loss_momentum,
            ),
        )

        # =========================================================
        # PROGRESSIVE SH
        # =========================================================

        self.use_progressive_sh = bool(
            optimization.get(
                "progressive_sh",
                True,
            )
        )

        self.sh_degree = int(
            optimization.get(
                "sh_degree",
                3,
            )
        )

        self.sh_start_degree = int(
            optimization.get(
                "sh_start_degree",
                0,
            )
        )

        self.sh_degree_1_iter = int(
            optimization.get(
                "sh_degree_1_iter",
                1000,
            )
        )

        self.sh_degree_2_iter = int(
            optimization.get(
                "sh_degree_2_iter",
                2000,
            )
        )

        self.sh_degree_3_iter = int(
            optimization.get(
                "sh_degree_3_iter",
                4000,
            )
        )

        if self.sh_degree < 0:

            raise ValueError(
                "sh_degree must be >= 0."
            )

        if self.sh_degree > 3:

            raise ValueError(
                "This trainer supports SH degree up to 3."
            )

        self.sh_start_degree = max(
            0,
            min(
                self.sh_start_degree,
                self.sh_degree,
            ),
        )

        # =========================================================
        # DETAIL LOSS SETTINGS
        # =========================================================

        self.detail_downsample = int(
            optimization.get(
                "detail_downsample",
                2,
            )
        )

        self.detail_blur_kernel = int(
            optimization.get(
                "detail_blur_kernel",
                3,
            )
        )

        # Keep these values valid.
        self.detail_downsample = max(
            self.detail_downsample,
            1,
        )

        self.detail_blur_kernel = max(
            self.detail_blur_kernel,
            1,
        )

        # =========================================================
        # RENDERER
        # =========================================================

        self.renderer = GaussianRenderer(
            config
        )

        # =========================================================
        # GSPLAT DEFAULT STRATEGY
        # =========================================================
        #
        # DefaultStrategy performs:
        #
        #   - Gaussian duplication
        #   - Gaussian splitting
        #   - Gaussian pruning
        #   - opacity reset
        #
        # It modifies the Gaussian parameter tensors and their
        # corresponding optimizers in-place.
        #
        # Current gsplat expects the parameters and optimizers
        # to have matching keys.
        #
        # =========================================================

        gaussian_config = config.get(
            "gaussian",
            {}
        )

        # ---------------------------------------------------------
        # AbsGS
        # ---------------------------------------------------------

        self.absgrad = bool(
            gaussian_config.get(
                "absgrad",
                optimization.get(
                    "absgrad",
                    True,
                ),
            )
        )

        # ---------------------------------------------------------
        # Strategy
        # ---------------------------------------------------------

        self.strategy = DefaultStrategy(

            prune_opa=float(
                optimization.get(
                    "prune_opa",
                    0.005,
                )
            ),

            grow_grad2d=float(
                optimization.get(
                    "grow_grad2d",
                    0.0008
                    if self.absgrad
                    else 0.0002,
                )
            ),

            grow_scale3d=float(
                optimization.get(
                    "grow_scale3d",
                    0.01,
                )
            ),

            grow_scale2d=float(
                optimization.get(
                    "grow_scale2d",
                    0.05,
                )
            ),

            prune_scale3d=float(
                optimization.get(
                    "prune_scale3d",
                    0.10,
                )
            ),

            prune_scale2d=float(
                optimization.get(
                    "prune_scale2d",
                    0.15,
                )
            ),

            refine_scale2d_stop_iter=int(
                optimization.get(
                    "refine_scale2d_stop_iter",
                    0,
                )
            ),

            refine_start_iter=int(
                optimization.get(
                    "refine_start_iter",
                    100,
                )
            ),

            refine_stop_iter=int(
                optimization.get(
                    "refine_stop_iter",
                    18000,
                )
            ),

            refine_every=int(
                optimization.get(
                    "refine_every",
                    100,
                )
            ),

            reset_every=int(
                optimization.get(
                    "reset_every",
                    3000,
                )
            ),

            pause_refine_after_reset=int(
                optimization.get(
                    "pause_refine_after_reset",
                    0,
                )
            ),

            absgrad=self.absgrad,

            revised_opacity=bool(
                optimization.get(
                    "revised_opacity",
                    False,
                )
            ),

            verbose=False,

            key_for_gradient="means2d",
        )

        # =========================================================
        # STRATEGY STATE
        # =========================================================

        self.strategy_state = (
            self.strategy.initialize_state()
        )

        # =========================================================
        # FRAME STATE
        # =========================================================

        self._frame_order = []

        self._frame_position = 0

        self._frame_losses = [
            1.0
            for _ in range(
                len(self.dataset)
            )
        ]

        self._frame_seen = [
            0
            for _ in range(
                len(self.dataset)
            )
        ]

        # =========================================================
        # CUDA SOBEL FILTERS
        # =========================================================

        self._sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(
            1,
            1,
            3,
            3,
        )

        self._sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(
            1,
            1,
            3,
            3,
        )

        # =========================================================
        # HELD-OUT EVALUATION SPLIT
        # =========================================================
        #
        # Nothing in this trainer previously measured quality on
        # a view the model wasn't trained on -- only training
        # loss on frames it's actively memorizing. A model can
        # have very low training loss while still looking soft
        # or wrong from angles it wasn't shown (classic
        # overfitting), and training loss alone cannot detect
        # that.
        #
        # Standard practice (matches the Mip-NeRF360 / original
        # 3DGS paper evaluation protocol): hold out every Nth
        # frame, by dataset order, as a test set that is NEVER
        # used for training, and periodically render it to
        # compute PSNR/SSIM as an objective quality signal.
        # =========================================================

        self.held_out_every = int(
            optimization.get(
                "held_out_every",
                8,
            )
        )

        self.held_out_every = max(
            self.held_out_every,
            2,
        )

        dataset_size = len(self.dataset)

        # Only bother holding out frames if there's enough data
        # that doing so won't cripple training itself.
        if dataset_size >= 10:

            self._held_out_indices = [
                index
                for index in range(dataset_size)
                if index % self.held_out_every == 0
            ]

        else:

            self._held_out_indices = []

            print(
                "Dataset too small "
                f"({dataset_size} frames) to hold out an "
                "evaluation split; training on all frames "
                "and skipping PSNR/SSIM evaluation."
            )

        self._held_out_set = set(
            self._held_out_indices
        )

        self._train_indices = [
            index
            for index in range(dataset_size)
            if index not in self._held_out_set
        ]

        if (
            self._held_out_indices
            and not self._train_indices
        ):

            raise RuntimeError(
                "held_out_every is too small: no frames "
                "remain for training after removing the "
                "held-out evaluation split."
            )

        if self._held_out_indices:

            print(
                f"Held-out evaluation frames: "
                f"{len(self._held_out_indices)} "
                f"/ {dataset_size} "
                f"(every {self.held_out_every}th frame, "
                "excluded from training)"
            )

    # =============================================================
    # FRAME SAMPLING
    # =============================================================

    def _next_frame(self):

        count = len(
            self._train_indices
        )

        if count == 0:

            raise RuntimeError(
                "Gaussian training dataset is empty."
            )

        # ---------------------------------------------------------
        # NORMAL RANDOM SHUFFLE
        # ---------------------------------------------------------

        if not self.adaptive_sampling:

            if (
                not self._frame_order
                or self._frame_position
                >= len(self._frame_order)
            ):

                self._frame_order = [
                    self._train_indices[position]
                    for position in torch.randperm(
                        count
                    ).tolist()
                ]

                self._frame_position = 0

            index = self._frame_order[
                self._frame_position
            ]

            self._frame_position += 1

            return (
                index,
                self.dataset[index],
            )

        # ---------------------------------------------------------
        # CREATE NEW SHUFFLED EPOCH
        # ---------------------------------------------------------

        if (
            not self._frame_order
            or self._frame_position
            >= len(self._frame_order)
        ):

            self._frame_order = [
                self._train_indices[position]
                for position in torch.randperm(
                    count
                ).tolist()
            ]

            self._frame_position = 0

        # ---------------------------------------------------------
        # HARD FRAME VS UNIFORM FRAME
        # ---------------------------------------------------------
        #
        # The previous implementation used:
        #
        #     if uniform:
        #         ...
        #
        #     if hard:
        #         ...
        #
        # which made the actual hard-frame probability:
        #
        #     (1 - uniform_probability)
        #     * hard_probability
        #
        # That is unintuitive.
        #
        # Here hard_frame_probability directly represents the
        # probability of choosing a hard frame.
        #
        # =========================================================

        if (
            random.random()
            < self.hard_frame_probability
            and sum(self._frame_seen) > 0
        ):

            losses = torch.tensor(
                [
                    self._frame_losses[index]
                    for index in self._train_indices
                ],
                dtype=torch.float32,
            )

            losses = torch.clamp(
                losses,
                min=1e-6,
            )

            probabilities = (
                losses
                / losses.sum()
            )

            selected = int(
                torch.multinomial(
                    probabilities,
                    1,
                ).item()
            )

            index = self._train_indices[
                selected
            ]

            return (
                index,
                self.dataset[index],
            )

        # ---------------------------------------------------------
        # UNIFORM FRAME
        # ---------------------------------------------------------

        index = self._frame_order[
            self._frame_position
        ]

        self._frame_position += 1

        return (
            index,
            self.dataset[index],
        )

    # =============================================================
    # FRAME LOSS UPDATE
    # =============================================================

    def _update_frame_loss(
        self,
        index,
        loss,
    ):

        value = float(
            loss
        )

        old = self._frame_losses[
            index
        ]

        momentum = (
            self.frame_loss_momentum
        )

        self._frame_losses[
            index
        ] = (
            momentum * old
            + (1.0 - momentum) * value
        )

        self._frame_seen[
            index
        ] += 1

    # =============================================================
    # POSITION LEARNING RATE
    # =============================================================

    def _update_position_learning_rate(
        self,
        iteration,
    ):

        if self.iterations <= 1:

            return

        # ---------------------------------------------------------
        # WARMUP
        # ---------------------------------------------------------

        if (
            self.position_warmup > 0
            and iteration <= self.position_warmup
        ):

            progress = (
                iteration
                / float(
                    self.position_warmup
                )
            )

            progress = max(
                0.0,
                min(
                    1.0,
                    progress,
                ),
            )

            # Start at 10% of base LR and reach 100%.
            multiplier = (
                0.10
                + 0.90 * progress
            )

        # ---------------------------------------------------------
        # EXPONENTIAL DECAY
        # ---------------------------------------------------------

        else:

            decay_start = max(
                self.position_warmup,
                1,
            )

            progress = (
                iteration
                - decay_start
            ) / float(
                max(
                    self.iterations
                    - decay_start,
                    1,
                )
            )

            progress = max(
                0.0,
                min(
                    1.0,
                    progress,
                ),
            )

            multiplier = (
                self.position_lr_final_factor
                ** progress
            )

        lr = (
            self.base_position_lr
            * multiplier
        )

        if "means" in self.optimizers:

            for group in self.optimizers[
                "means"
            ].param_groups:

                group["lr"] = lr

    # =============================================================
    # PROGRESSIVE SH DEGREE
    # =============================================================

    def _active_sh_degree(
        self,
        iteration,
    ):

        if not self.use_progressive_sh:

            return self.sh_degree

        if "shs" not in self.parameters:

            return 0

        degree = self.sh_start_degree

        if (
            self.sh_degree >= 1
            and iteration
            >= self.sh_degree_1_iter
        ):

            degree = max(
                degree,
                1,
            )

        if (
            self.sh_degree >= 2
            and iteration
            >= self.sh_degree_2_iter
        ):

            degree = max(
                degree,
                2,
            )

        if (
            self.sh_degree >= 3
            and iteration
            >= self.sh_degree_3_iter
        ):

            degree = max(
                degree,
                3,
            )

        return min(
            degree,
            self.sh_degree,
        )

    # =============================================================
    # SSIM LOSS
    # =============================================================

    def _ssim_loss(
        self,
        rendered,
        target,
    ):

        # [H, W, 3]
        #
        # ->
        #
        # [1, 3, H, W]

        rendered = (
            rendered
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        target = (
            target
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        # ---------------------------------------------------------
        # LOCAL MEANS
        # ---------------------------------------------------------

        mu_x = F.avg_pool2d(
            rendered,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        mu_y = F.avg_pool2d(
            target,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ---------------------------------------------------------
        # LOCAL VARIANCE
        # ---------------------------------------------------------

        sigma_x = (
            F.avg_pool2d(
                rendered * rendered,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            - mu_x * mu_x
        )

        sigma_y = (
            F.avg_pool2d(
                target * target,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            - mu_y * mu_y
        )

        # ---------------------------------------------------------
        # LOCAL COVARIANCE
        # ---------------------------------------------------------

        sigma_xy = (
            F.avg_pool2d(
                rendered * target,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            - mu_x * mu_y
        )

        # ---------------------------------------------------------
        # SSIM CONSTANTS
        # ---------------------------------------------------------

        c1 = 0.01 ** 2

        c2 = 0.03 ** 2

        numerator = (
            (
                2.0 * mu_x * mu_y
                + c1
            )
            * (
                2.0 * sigma_xy
                + c2
            )
        )

        denominator = (
            (
                mu_x * mu_x
                + mu_y * mu_y
                + c1
            )
            * (
                sigma_x
                + sigma_y
                + c2
            )
        )

        ssim = (
            numerator
            / (
                denominator
                + 1e-8
            )
        )

        # Numerical safety.
        ssim = torch.clamp(
            ssim,
            -1.0,
            1.0,
        )

        return (
            1.0
            - ssim.mean()
        )

    # =============================================================
    # EDGE / DETAIL LOSS
    # =============================================================

    def _edge_loss(
        self,
        rendered,
        target,
    ):

        rendered = (
            rendered
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        target = (
            target
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        # ---------------------------------------------------------
        # RGB -> GRAYSCALE
        # ---------------------------------------------------------

        rendered_gray = (
            0.299 * rendered[:, 0:1]
            + 0.587 * rendered[:, 1:2]
            + 0.114 * rendered[:, 2:3]
        )

        target_gray = (
            0.299 * target[:, 0:1]
            + 0.587 * target[:, 1:2]
            + 0.114 * target[:, 2:3]
        )

        # ---------------------------------------------------------
        # SOBEL
        # ---------------------------------------------------------

        render_x = F.conv2d(
            rendered_gray,
            self._sobel_x,
            padding=1,
        )

        render_y = F.conv2d(
            rendered_gray,
            self._sobel_y,
            padding=1,
        )

        target_x = F.conv2d(
            target_gray,
            self._sobel_x,
            padding=1,
        )

        target_y = F.conv2d(
            target_gray,
            self._sobel_y,
            padding=1,
        )

        # ---------------------------------------------------------
        # GRADIENT MAGNITUDE
        # ---------------------------------------------------------

        render_edges = torch.sqrt(
            render_x * render_x
            + render_y * render_y
            + 1e-6
        )

        target_edges = torch.sqrt(
            target_x * target_x
            + target_y * target_y
            + 1e-6
        )

        return F.l1_loss(
            render_edges,
            target_edges,
        )

    # =============================================================
    # MULTI-SCALE LOSS
    # =============================================================

    def _multiscale_loss(
        self,
        rendered,
        target,
    ):

        rendered = (
            rendered
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        target = (
            target
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
        )

        # ---------------------------------------------------------
        # FULL RESOLUTION
        # ---------------------------------------------------------

        full_loss = F.l1_loss(
            rendered,
            target,
        )

        # ---------------------------------------------------------
        # HALF RESOLUTION
        # ---------------------------------------------------------

        if (
            rendered.shape[-1] >= 4
            and rendered.shape[-2] >= 4
        ):

            rendered_half = F.avg_pool2d(
                rendered,
                kernel_size=2,
                stride=2,
            )

            target_half = F.avg_pool2d(
                target,
                kernel_size=2,
                stride=2,
            )

            half_loss = F.l1_loss(
                rendered_half,
                target_half,
            )

        else:

            half_loss = full_loss

        # ---------------------------------------------------------
        # QUARTER RESOLUTION
        # ---------------------------------------------------------

        if (
            rendered.shape[-1] >= 8
            and rendered.shape[-2] >= 8
        ):

            rendered_quarter = F.avg_pool2d(
                rendered,
                kernel_size=4,
                stride=4,
            )

            target_quarter = F.avg_pool2d(
                target,
                kernel_size=4,
                stride=4,
            )

            quarter_loss = F.l1_loss(
                rendered_quarter,
                target_quarter,
            )

        else:

            quarter_loss = half_loss

        return (
            0.50 * full_loss
            + 0.30 * half_loss
            + 0.20 * quarter_loss
        )

    # =============================================================
    # TOTAL LOSS
    # =============================================================

    def _loss(
        self,
        rendered,
        target,
    ):

        # ---------------------------------------------------------
        # L1
        # ---------------------------------------------------------

        l1 = torch.abs(
            rendered - target
        ).mean()

        # ---------------------------------------------------------
        # SSIM
        # ---------------------------------------------------------

        ssim = self._ssim_loss(
            rendered,
            target,
        )

        # ---------------------------------------------------------
        # EDGE
        # ---------------------------------------------------------

        edge = self._edge_loss(
            rendered,
            target,
        )

        # ---------------------------------------------------------
        # MULTI-SCALE
        # ---------------------------------------------------------

        multiscale = (
            self._multiscale_loss(
                rendered,
                target,
            )
        )

        # ---------------------------------------------------------
        # TOTAL
        # ---------------------------------------------------------

        total = (
            self.lambda_l1 * l1
            + self.lambda_ssim * ssim
            + self.lambda_edge * edge
            + self.lambda_multiscale
            * multiscale
        )

        return (
            total,
            l1.detach(),
            ssim.detach(),
            edge.detach(),
            multiscale.detach(),
        )

    # =============================================================
    # STRATEGY PARAMETERS
    # =============================================================

    def _strategy_parameters(self):

        # =========================================================
        # IMPORTANT
        # =========================================================
        #
        # DefaultStrategy requires:
        #
        # means
        # scales
        # quats
        # opacities
        #
        # Additional per-Gaussian parameters such as "shs" are
        # also supported.
        #
        # =========================================================

        result = {

            "means": self.parameters[
                "means"
            ],

            "scales": self.parameters[
                "scales"
            ],

            "quats": self.parameters[
                "quats"
            ],

            "opacities": self.parameters[
                "opacities"
            ],
        }

        if "colors" in self.parameters:

            result[
                "colors"
            ] = self.parameters[
                "colors"
            ]

        if "shs" in self.parameters:

            result[
                "shs"
            ] = self.parameters[
                "shs"
            ]

        return result

    # =============================================================
    # HELD-OUT EVALUATION
    # =============================================================

    @torch.no_grad()
    def _evaluate_held_out(
        self,
        active_sh_degree,
    ):
        """
        Render every held-out (never-trained-on) frame and
        compute PSNR/SSIM against the real photo. This is the
        only objective, numeric signal in this pipeline for
        "does this actually look right from angles the model
        didn't memorize" -- training loss alone cannot tell you
        that.

        Returns None if no held-out split exists (see __init__).
        """

        if not self._held_out_indices:
            return None

        psnr_values = []
        ssim_values = []

        for index in self._held_out_indices:

            frame = self.dataset[index]

            target = frame["image"]

            result = self.renderer.render(
                self.parameters,
                frame,
                active_sh_degree=active_sh_degree,
            )

            rendered = torch.clamp(
                result["image"],
                0.0,
                1.0,
            )

            mse = torch.mean(
                (rendered - target) ** 2
            )

            mse_value = float(mse.clamp_min(1e-10))

            psnr = -10.0 * (
                torch.log10(
                    torch.tensor(mse_value)
                ).item()
            )

            ssim_value = float(
                1.0
                - self._ssim_loss(
                    rendered,
                    target,
                )
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

        return {
            "psnr": sum(psnr_values) / len(psnr_values),
            "ssim": sum(ssim_values) / len(ssim_values),
            "count": len(self._held_out_indices),
        }

    # =============================================================
    # CHECKPOINT
    # =============================================================

    def _save_checkpoint(
        self,
        output_path,
        iteration,
        loss,
        eval_metrics=None,
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_data = {
            "iteration": int(
                iteration
            ),

            "parameters": {
                key: value.detach().cpu()
                for key, value
                in self.parameters.items()
            },

            "loss": float(
                loss
            ),

            "gaussians": int(
                self.parameters[
                    "means"
                ].shape[0]
            ),
        }

        if eval_metrics is not None:

            checkpoint_data[
                "held_out_psnr"
            ] = eval_metrics["psnr"]

            checkpoint_data[
                "held_out_ssim"
            ] = eval_metrics["ssim"]

        torch.save(
            checkpoint_data,
            output_path,
        )

    # =============================================================
    # TRAIN
    # =============================================================

    def train(
        self,
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
        # STRATEGY SANITY CHECK
        # =========================================================

        strategy_parameters = self.parameters

        self.strategy.check_sanity(
            strategy_parameters,
            self.optimizers,
        )

        # =========================================================
        # TRAINING STATE
        # =========================================================

        last_loss = None

        initial_gaussian_count = int(
            self.parameters[
                "means"
            ].shape[0]
        )

        previous_gaussian_count = (
            initial_gaussian_count
        )

        # =========================================================
        # PROGRESS BAR
        # =========================================================

        progress = tqdm(
            range(
                1,
                self.iterations + 1,
            ),
            desc="Gaussian training",
            unit="iter",
        )

        # =========================================================
        # TRAINING LOOP
        # =========================================================

        for iteration in progress:

            # -----------------------------------------------------
            # UPDATE POSITION LEARNING RATE
            # -----------------------------------------------------

            self._update_position_learning_rate(
                iteration
            )

            # -----------------------------------------------------
            # SELECT FRAME
            # -----------------------------------------------------

            (
                frame_index,
                frame,
            ) = self._next_frame()

            # -----------------------------------------------------
            # ZERO GRADIENTS
            # -----------------------------------------------------

            for optimizer in (
                self.optimizers.values()
            ):

                optimizer.zero_grad(
                    set_to_none=True
                )

            # -----------------------------------------------------
            # ACTIVE SH DEGREE
            # -----------------------------------------------------

            active_sh_degree = (
                self._active_sh_degree(
                    iteration
                )
            )

            # -----------------------------------------------------
            # RENDER
            # -----------------------------------------------------

            info = self.renderer.render(
                self.parameters,
                frame,
                active_sh_degree=active_sh_degree,
            )

            # -----------------------------------------------------
            # VALIDATE RENDER OUTPUT
            # -----------------------------------------------------

            if "image" not in info:

                raise RuntimeError(
                    "Renderer did not return 'image'."
                )

            if "means2d" not in info:

                raise RuntimeError(
                    "Renderer did not return 'means2d'."
                )

            means2d = info[
                "means2d"
            ]

            if means2d is None:

                raise RuntimeError(
                    "Renderer returned means2d=None."
                )

            if not means2d.requires_grad:

                raise RuntimeError(
                    "Renderer means2d does not require gradients."
                )



            # -----------------------------------------------------
            # STRATEGY INFO
            # -----------------------------------------------------
            #
            # Keep every gsplat metadata field.
            #
            # DefaultStrategy needs fields such as:
            #
            #   means2d
            #   radii
            #   width
            #   height
            #   n_cameras
            #   gaussian_ids
            #
            # depending on packed/unpacked rasterization.
            #
            # Therefore we must NOT reconstruct this dictionary
            # manually.
            #
            # =====================================================

            strategy_info = info

            # -----------------------------------------------------
            # TARGET IMAGE
            # -----------------------------------------------------

            rendered = info[
                "image"
            ]

            target = frame[
                "image"
            ]

            if not torch.is_tensor(
                target
            ):

                raise TypeError(
                    "frame['image'] must be a torch.Tensor."
                )

            if target.device != self.device:

                target = target.to(
                    self.device,
                    non_blocking=True,
                )

            target = target.float()

            # -----------------------------------------------------
            # NORMALIZE TARGET
            # -----------------------------------------------------

            if target.numel() > 0:

                if float(
                    target.detach().max()
                ) > 1.0:

                    target = (
                        target
                        / 255.0
                    )

            target = torch.clamp(
                target,
                0.0,
                1.0,
            )

            # -----------------------------------------------------
            # TARGET SHAPE
            # -----------------------------------------------------

            if target.ndim != 3:

                raise ValueError(
                    "Target image must have shape [H, W, 3]."
                )

            if target.shape[-1] != 3:

                raise ValueError(
                    "Target image must have 3 color channels."
                )

            if rendered.ndim != 3:

                raise ValueError(
                    "Rendered image must have shape [H, W, 3]."
                )

            if rendered.shape[-1] != 3:

                raise ValueError(
                    "Rendered image must have 3 color channels."
                )

            if (
                rendered.shape[0] != target.shape[0]
                or rendered.shape[1] != target.shape[1]
            ):

                raise ValueError(
                    "Rendered and target image resolutions differ. "
                    f"Rendered={tuple(rendered.shape)}, "
                    f"Target={tuple(target.shape)}."
                )

            # -----------------------------------------------------
            # PRE-BACKWARD STRATEGY
            # -----------------------------------------------------
            #
            # Current gsplat DefaultStrategy explicitly expects
            # this before loss.backward().
            #
            # It retains the gradient of means2d so that the
            # post-backward densification step can inspect it.
            #
            # =====================================================

            self.strategy.step_pre_backward(
                strategy_parameters,
                self.optimizers,
                self.strategy_state,
                iteration,
                strategy_info,
            )

            # -----------------------------------------------------
            # LOSS
            # -----------------------------------------------------

            (
                loss,
                l1,
                ssim,
                edge,
                multiscale,
            ) = self._loss(
                rendered,
                target,
            )

            # -----------------------------------------------------
            # CHECK LOSS
            # -----------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite Gaussian training loss detected at "
                    f"iteration {iteration}: "
                    f"{float(loss.detach())}"
                )

            # -----------------------------------------------------
            # BACKWARD
            # -----------------------------------------------------

            loss.backward()

            # -----------------------------------------------------
            # POST-BACKWARD STRATEGY
            # -----------------------------------------------------
            #
            # DefaultStrategy performs densification/pruning here.
            #
            # IMPORTANT:
            #
            # This must happen AFTER backward and BEFORE optimizer
            # step, matching the gsplat training workflow.
            #
            # =====================================================

            self.strategy.step_post_backward(
                strategy_parameters,
                self.optimizers,
                self.strategy_state,
                iteration,
                strategy_info,
                packed=True,
            )

            # -----------------------------------------------------
            # OPTIMIZER STEP
            # -----------------------------------------------------

            for optimizer in (
                self.optimizers.values()
            ):

                optimizer.step()

            # -----------------------------------------------------
            # FRAME DIFFICULTY
            # -----------------------------------------------------

            self._update_frame_loss(
                frame_index,
                loss.detach(),
            )

            # -----------------------------------------------------
            # GAUSSIAN COUNT
            # -----------------------------------------------------

            current_gaussian_count = int(
                self.parameters[
                    "means"
                ].shape[0]
            )

            gaussian_change = (
                current_gaussian_count
                - previous_gaussian_count
            )

            previous_gaussian_count = (
                current_gaussian_count
            )

            last_loss = loss.detach()

            # -----------------------------------------------------
            # CUDA MEMORY MANAGEMENT
            # -----------------------------------------------------

            if gaussian_change != 0:

                torch.cuda.empty_cache()

            # -----------------------------------------------------
            # CURRENT POSITION LR
            # -----------------------------------------------------

            current_position_lr = (
                self.base_position_lr
            )

            if "means" in self.optimizers:

                current_position_lr = (
                    self.optimizers[
                        "means"
                    ].param_groups[
                        0
                    ][
                        "lr"
                    ]
                )

            # -----------------------------------------------------
            # PROGRESS BAR
            # -----------------------------------------------------

            progress.set_postfix(

                loss=f"{float(last_loss):.6f}",

                l1=f"{float(l1):.4f}",

                ssim=f"{float(ssim):.4f}",

                edge=f"{float(edge):.4f}",

                multi=f"{float(multiscale):.4f}",

                sh=active_sh_degree,

                lr=f"{current_position_lr:.2e}",

                gaussians=current_gaussian_count,

                delta=gaussian_change,
            )

            # -----------------------------------------------------
            # CHECKPOINT
            # -----------------------------------------------------

            if (
                iteration
                % self.checkpoint_interval
                == 0
            ):

                eval_metrics = (
                    self._evaluate_held_out(
                        active_sh_degree
                    )
                )

                self._save_checkpoint(
                    output_path,
                    iteration,
                    last_loss,
                    eval_metrics=eval_metrics,
                )

                print()

                print(
                    "=" * 70
                )

                print(
                    f"CHECKPOINT — ITERATION {iteration}"
                )

                print(
                    f"Gaussians:        "
                    f"{current_gaussian_count}"
                )

                print(
                    f"Loss:              "
                    f"{float(last_loss):.6f}"
                )

                print(
                    f"L1:                "
                    f"{float(l1):.6f}"
                )

                print(
                    f"SSIM:              "
                    f"{float(ssim):.6f}"
                )

                print(
                    f"Edge:              "
                    f"{float(edge):.6f}"
                )

                print(
                    f"Multi-scale:       "
                    f"{float(multiscale):.6f}"
                )

                print(
                    f"Active SH degree:  "
                    f"{active_sh_degree}"
                )

                print(
                    f"Position LR:       "
                    f"{current_position_lr:.8f}"
                )

                if eval_metrics is not None:

                    print(
                        f"Held-out PSNR:     "
                        f"{eval_metrics['psnr']:.2f} dB "
                        f"({eval_metrics['count']} frames, "
                        "never trained on)"
                    )

                    print(
                        f"Held-out SSIM:     "
                        f"{eval_metrics['ssim']:.4f}"
                    )

                print(
                    "=" * 70
                )

        # =========================================================
        # FINAL VALIDATION
        # =========================================================

        if last_loss is None:

            raise RuntimeError(
                "Gaussian training completed without "
                "performing any iterations."
            )

        # =========================================================
        # FINAL CHECKPOINT
        # =========================================================

        final_count = int(
            self.parameters[
                "means"
            ].shape[0]
        )

        final_eval_metrics = (
            self._evaluate_held_out(
                self._active_sh_degree(
                    self.iterations
                )
            )
        )

        self._save_checkpoint(
            output_path,
            self.iterations,
            last_loss,
            eval_metrics=final_eval_metrics,
        )

        # =========================================================
        # TRAINING REPORT
        # =========================================================

        print()

        print(
            "=" * 70
        )

        print(
            "GAUSSIAN TRAINING COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Iterations:         "
            f"{self.iterations}"
        )

        print(
            f"Initial Gaussians:  "
            f"{initial_gaussian_count}"
        )

        print(
            f"Final Gaussians:    "
            f"{final_count}"
        )

        print(
            f"Gaussian change:    "
            f"{final_count - initial_gaussian_count}"
        )

        print(
            f"Final loss:         "
            f"{float(last_loss):.6f}"
        )

        print(
            f"Final SH degree:    "
            f"{self._active_sh_degree(self.iterations)}"
        )

        print(
            f"Checkpoint:         "
            f"{output_path}"
        )

        if final_eval_metrics is not None:

            print(
                f"Held-out PSNR:      "
                f"{final_eval_metrics['psnr']:.2f} dB "
                f"({final_eval_metrics['count']} frames)"
            )

            print(
                f"Held-out SSIM:      "
                f"{final_eval_metrics['ssim']:.4f}"
            )

        print(
            "=" * 70
        )

        return {

            "iterations":
                self.iterations,

            "final_loss":
                float(last_loss),

            "initial_gaussians":
                initial_gaussian_count,

            "gaussians":
                final_count,

            "checkpoint":
                str(output_path),

            "held_out_psnr":
                final_eval_metrics["psnr"]
                if final_eval_metrics is not None
                else None,

            "held_out_ssim":
                final_eval_metrics["ssim"]
                if final_eval_metrics is not None
                else None,
        }