from pathlib import Path

from src.gaussian.dataset import GaussianDataset
from src.gaussian.initializer import GaussianInitializer
from src.gaussian.optimizer import GaussianOptimizer
from src.gaussian.trainer import GaussianTrainer


class GaussianPipeline:
    def __init__(
        self,
        sparse_path,
        image_path,
        initialization_path,
        checkpoint_path,
        config,
    ):
        self.sparse_path = Path(
            sparse_path
        )

        self.image_path = Path(
            image_path
        )

        self.initialization_path = Path(
            initialization_path
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.config = config

        # ---------------------------------------------------------
        # BASIC VALIDATION
        # ---------------------------------------------------------

        if not self.sparse_path.exists():
            raise FileNotFoundError(
                "Sparse reconstruction not found: "
                f"{self.sparse_path}"
            )

        if not self.image_path.exists():
            raise FileNotFoundError(
                "Gaussian training image directory not found: "
                f"{self.image_path}"
            )

        self.initialization_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------

    def initialize(self):
        print()
        print(
            "GAUSSIAN INITIALIZATION"
        )
        print(
            "-" * 70
        )

        initializer = GaussianInitializer(
            self.config
        )

        initialization = (
            initializer.initialize(
                self.sparse_path,
                self.initialization_path,
            )
        )

        if not self.initialization_path.exists():
            raise RuntimeError(
                "Gaussian initialization completed "
                "but the initialization file was not created."
            )

        if initialization["points"] <= 0:
            raise RuntimeError(
                "Gaussian initialization produced "
                "zero valid points."
            )

        print(
            "Initial Gaussian points:",
            initialization["points"],
        )

        print(
            "Mean Gaussian scale:",
            initialization["mean_scale"],
        )

        print(
            "Mean track length:",
            initialization["mean_track_length"],
        )

        print(
            "Mean reprojection error:",
            initialization[
                "mean_reprojection_error"
            ],
        )

        return initialization

    # -------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------

    def create_dataset(self):
        print()
        print(
            "GAUSSIAN DATASET"
        )
        print(
            "-" * 70
        )

        dataset = GaussianDataset(
            self.sparse_path,
            self.image_path,
            self.config,
        )

        if len(dataset) < 3:
            raise RuntimeError(
                "Gaussian training requires at least "
                "3 posed training images."
            )

        print(
            "Training frames:",
            len(dataset),
        )

        return dataset

    # -------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------

    def create_optimizer(
        self,
    ):
        print()
        print(
            "GAUSSIAN OPTIMIZER"
        )
        print(
            "-" * 70
        )

        optimizer_factory = (
            GaussianOptimizer(
                self.config
            )
        )

        (
            parameters,
            optimizers,
        ) = optimizer_factory.initialize(
            self.initialization_path
        )

        if not parameters:
            raise RuntimeError(
                "Gaussian optimizer received "
                "an empty parameter set."
            )

        if not optimizers:
            raise RuntimeError(
                "Gaussian optimizer created "
                "no optimizers."
            )

        print(
            "Gaussian parameters:",
            len(parameters),
        )

        print(
            "Optimizers:",
            len(optimizers),
        )

        return (
            parameters,
            optimizers,
        )

    # -------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------

    def train(self):
        print()
        print(
            "GAUSSIAN TRAINING"
        )
        print(
            "-" * 70
        )

        if not self.initialization_path.exists():
            raise FileNotFoundError(
                "Gaussian initialization file does not exist: "
                f"{self.initialization_path}"
            )

        dataset = self.create_dataset()

        (
            parameters,
            optimizers,
        ) = self.create_optimizer()

        trainer = GaussianTrainer(
            dataset,
            parameters,
            optimizers,
            self.config,
        )

        training = trainer.train(
            self.checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "Gaussian training completed "
                "but no checkpoint was created: "
                f"{self.checkpoint_path}"
            )

        print()
        print(
            "Gaussian checkpoint:",
            self.checkpoint_path,
        )

        return training

    # -------------------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------------------

    def run(self):
        print()
        print(
            "=" * 70
        )
        print(
            "GAUSSIAN PIPELINE"
        )
        print(
            "=" * 70
        )

        # ---------------------------------------------------------
        # INITIALIZATION
        # ---------------------------------------------------------

        initialization = (
            self.initialize()
        )

        # ---------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------

        training = (
            self.train()
        )

        # ---------------------------------------------------------
        # FINAL VALIDATION
        # ---------------------------------------------------------

        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "Gaussian pipeline failed: "
                "final checkpoint does not exist."
            )

        print()
        print(
            "=" * 70
        )
        print(
            "GAUSSIAN PIPELINE COMPLETE"
        )
        print(
            "=" * 70
        )

        print(
            "Initialized points:",
            initialization[
                "points"
            ],
        )

        print(
            "Checkpoint:",
            self.checkpoint_path,
        )

        return {
            "initialization":
                initialization,

            "training":
                training,

            "checkpoint":
                str(
                    self.checkpoint_path
                ),

            "status":
                "complete",
        }