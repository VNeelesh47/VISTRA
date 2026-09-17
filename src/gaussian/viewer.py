from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# SIH26158 — GAUSSIAN MODEL VIEWER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLY_PATH = (
    PROJECT_ROOT
    / "output"
    / "gaussian_model.ply"
)

SIMPLE_PLY_PATH = (
    PROJECT_ROOT
    / "output"
    / "point_cloud.ply"
)


def load_point_cloud(path):

    print()
    print("=" * 70)
    print("LOADING 3D MODEL")
    print("=" * 70)

    print("File:", path)

    if not path.exists():
        raise FileNotFoundError(
            f"PLY file not found:\n{path}"
        )

    point_cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "PLY contains no points."
        )

    print(
        "Points:",
        len(point_cloud.points)
    )

    return point_cloud


def prepare_point_cloud(point_cloud):

    points = np.asarray(
        point_cloud.points,
        dtype=np.float64,
    )

    print()
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print(
        "Point count:",
        len(points)
    )

    print(
        "Minimum:",
        points.min(axis=0)
    )

    print(
        "Maximum:",
        points.max(axis=0)
    )

    if point_cloud.has_colors():

        print(
            "Colors: available"
        )

    else:

        print(
            "PLY has no colors."
        )

        colors = np.full(
            (len(points), 3),
            0.7,
            dtype=np.float64,
        )

        point_cloud.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

    return point_cloud


def create_coordinate_frame(point_cloud):

    points = np.asarray(
        point_cloud.points
    )

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    size = np.linalg.norm(
        maximum - minimum
    )

    if size <= 0:
        size = 1.0

    center = (
        minimum + maximum
    ) / 2.0

    axis_size = size * 0.15

    coordinate_frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=axis_size,
            origin=center,
        )
    )

    return coordinate_frame


def create_visualizer(point_cloud):

    print()
    print("=" * 70)
    print("OPENING VIEWER")
    print("=" * 70)

    coordinate_frame = (
        create_coordinate_frame(
            point_cloud
        )
    )

    visualizer = (
        o3d.visualization.Visualizer()
    )

    visualizer.create_window(
        window_name=(
            "SIH26158 - Gaussian "
            "3D Reconstruction"
        ),
        width=1400,
        height=900,
    )

    visualizer.add_geometry(
        point_cloud
    )

    visualizer.add_geometry(
        coordinate_frame
    )

    render_option = (
        visualizer.get_render_option()
    )

    render_option.point_size = 3.0

    render_option.background_color = (
        np.asarray(
            [1.0, 1.0, 1.0]
        )
    )

    view_control = (
        visualizer.get_view_control()
    )

    view_control.set_front(
        [0.0, 0.0, -1.0]
    )

    view_control.set_up(
        [0.0, -1.0, 0.0]
    )

    view_control.set_zoom(
        0.8
    )

    print()
    print("Controls:")
    print("  Left mouse   : Rotate")
    print("  Right mouse  : Pan")
    print("  Scroll       : Zoom")
    print("  Mouse wheel  : Zoom")
    print("  R            : Reset")
    print("  Q / ESC      : Exit")
    print()

    visualizer.run()

    visualizer.destroy_window()


def main():

    print()
    print("=" * 70)
    print("SIH26158 - GAUSSIAN 3D MODEL VIEWER")
    print("=" * 70)

    if PLY_PATH.exists():

        selected_path = PLY_PATH

    elif SIMPLE_PLY_PATH.exists():

        print()
        print(
            "Gaussian PLY not found."
        )

        print(
            "Using simple point-cloud PLY."
        )

        selected_path = SIMPLE_PLY_PATH

    else:

        raise FileNotFoundError(
            "\nNo PLY model found.\n\n"
            "Expected one of:\n"
            f"  {PLY_PATH}\n"
            f"  {SIMPLE_PLY_PATH}\n"
        )

    point_cloud = load_point_cloud(
        selected_path
    )

    point_cloud = prepare_point_cloud(
        point_cloud
    )

    create_visualizer(
        point_cloud
    )

    print()
    print("=" * 70)
    print("VIEWER CLOSED")
    print("=" * 70)


if __name__ == "__main__":
    main()