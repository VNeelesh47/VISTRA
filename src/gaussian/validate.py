import argparse
from pathlib import Path

import torch
import yaml

from src.gaussian.dataset import GaussianDataset
from src.gaussian.initializer import GaussianInitializer
from src.gaussian.optimizer import GaussianOptimizer
from src.gaussian.renderer import GaussianRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--sparse",
        required=True,
    )
    parser.add_argument(
        "--images",
        required=True,
    )
    parser.add_argument(
        "--initialization",
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    initializer = GaussianInitializer(
        config
    )

    initialization = initializer.initialize(
        args.sparse,
        args.initialization,
    )

    print(
        "Initialized Gaussians:",
        initialization["points"],
    )

    dataset = GaussianDataset(
        args.sparse,
        args.images,
        config,
    )

    print(
        "Training images:",
        len(dataset),
    )

    optimizer_factory = GaussianOptimizer(
        config
    )

    parameters, optimizers = (
        optimizer_factory.initialize(
            args.initialization
        )
    )

    print(
        "Means:",
        tuple(parameters["means"].shape),
    )

    print(
        "Scales:",
        tuple(parameters["scales"].shape),
    )

    print(
        "Rotations:",
        tuple(parameters["rotations"].shape),
    )

    print(
        "Opacities:",
        tuple(parameters["opacities"].shape),
    )

    print(
        "Colors:",
        tuple(parameters["colors"].shape),
    )

    renderer = GaussianRenderer(
        config
    )

    frame = dataset[0]

    result = renderer.render(
        parameters,
        frame,
    )

    rendered = result["image"]

    print(
        "Rendered image:",
        tuple(rendered.shape),
    )

    print(
        "Alpha:",
        tuple(result["alpha"].shape),
    )

    if not torch.isfinite(
        rendered
    ).all():
        raise RuntimeError(
            "Rendered image contains NaN or Inf."
        )

    loss = (
        torch.abs(
            rendered
            - frame["image"]
        ).mean()
    )

    print(
        "Test loss:",
        float(loss),
    )

    loss.backward()

    print(
        "Backward pass: OK"
    )

    for name, parameter in parameters.items():
        if parameter.grad is None:
            print(
                f"Gradient missing: {name}"
            )
        elif not torch.isfinite(
            parameter.grad
        ).all():
            raise RuntimeError(
                f"Invalid gradient: {name}"
            )
        else:
            print(
                f"Gradient OK: {name}"
            )

    for optimizer in optimizers.values():
        optimizer.step()

    print(
        "Optimizer step: OK"
    )

    torch.cuda.synchronize()

    print(
        "Gaussian pipeline validation: PASSED"
    )


if __name__ == "__main__":
    main()