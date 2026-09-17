from pathlib import Path
import argparse

from src.gaussian.viewer import SIH26158Viewer


def main():
    parser = argparse.ArgumentParser(
        description="SIH26158 Gaussian 3D Viewer"
    )

    parser.add_argument(
        "--input",
        default=r"output\gaussian_model.ply",
    )

    parser.add_argument(
        "--sparse",
        default=r"reconstruction\sparse\0",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1400,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=900,
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    ply_path = (
        project_root / args.input
    )

    sparse_path = (
        project_root / args.sparse
    )

    print()
    print("=" * 70)
    print("SIH26158 — GAUSSIAN 3D VIEWER")
    print("=" * 70)

    print(
        f"Gaussian model : {ply_path}"
    )

    print(
        f"Sparse model   : {sparse_path}"
    )

    print()

    viewer = SIH26158Viewer(
        ply_path=ply_path,
        sparse_path=sparse_path,
        width=args.width,
        height=args.height,
    )

    viewer.run()


if __name__ == "__main__":
    main()