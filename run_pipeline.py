import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

from src.config.settings import load_config
from src.video.sampler import VideoSampler
from src.frames.filter import FrameFilter
from src.frames.selector import KeyframeSelector
from src.features.extractor import FeatureExtractionPipeline
from src.matching.pipeline import MatchingPipeline
from src.sfm.camera import CameraBuilder
from src.sfm.reconstruction_runner import ReconstructionRunner
from src.gaussian.initializer import GaussianInitializer
from src.pipeline.manifest import PipelineManifest


def find_video(input_dir):
    input_dir = Path(input_dir)

    extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }

    videos = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not videos:
        raise FileNotFoundError(
            f"No video files found in: {input_dir}"
        )

    if len(videos) > 1:
        print("Multiple videos found:")
        for index, video in enumerate(videos):
            print(f"  [{index}] {video}")

        print(
            "\nUsing the first video. "
            "Pass --video to select another one."
        )

    return videos[0]


def load_requested_video(video_argument, input_dir):
    if video_argument:
        video_path = Path(video_argument)

        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path

        video_path = video_path.resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        return video_path

    return find_video(input_dir)


def save_json(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )



def export_gaussian_ply(checkpoint_path, output_path):
    """Export trained Gaussian centers and colors as a viewer-compatible PLY."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Gaussian checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "parameters" in checkpoint:
        parameters = checkpoint["parameters"]
    else:
        parameters = checkpoint

    required = ["means", "colors"]
    missing = [key for key in required if key not in parameters]
    if missing:
        raise RuntimeError(
            "Gaussian checkpoint is missing required parameters: "
            + ", ".join(missing)
        )

    means = parameters["means"].detach().float().cpu().numpy()
    colors = parameters["colors"].detach().float().cpu().numpy()

    if means.ndim != 2 or means.shape[1] != 3:
        raise RuntimeError(f"Invalid Gaussian means shape: {means.shape}")

    # The trainer stores color as logits; convert back to RGB [0,255].
    colors = 1.0 / (1.0 + np.exp(-np.clip(colors, -30.0, 30.0)))
    colors = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)

    valid = np.isfinite(means).all(axis=1) & np.isfinite(colors).all(axis=1)
    means = means[valid]
    colors = colors[valid]

    if len(means) == 0:
        raise RuntimeError("Checkpoint contains no valid Gaussian points.")

    with output_path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(means)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for xyz, rgb in zip(means, colors):
            file.write(
                f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )

    return {
        "output": str(output_path),
        "vertices": int(len(means)),
        "size_mb": round(output_path.stat().st_size / (1024 * 1024), 3),
    }


def run_module(module_name, arguments):
    command = [sys.executable, "-m", module_name] + [str(x) for x in arguments]
    completed = subprocess.run(command, cwd=str(Path.cwd()), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Module '{module_name}' failed with exit code {completed.returncode}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="SIH26158 video-to-3D pipeline"
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Input video path. If omitted, the first video in input/videos is used.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the full configured iteration Gaussian training stage.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SIH26158 VIDEO → 3D PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    config = load_config()

    paths = config["paths"]

    input_dir = Path(paths["input"])
    output_dir = Path(paths["output"])
    frames_dir = Path(paths["frames"])
    selected_dir = Path(paths["selected"])
    features_dir = Path(paths["features"])
    matches_dir = Path(paths["matches"])
    reconstruction_dir = Path(
        paths["reconstruction"]
    )

    reconstruction_database = Path(
        config["reconstruction"]["database"]
    )

    if not reconstruction_database.is_absolute():
        reconstruction_database = (
            Path.cwd() / reconstruction_database
        )

    sparse_dir = Path(
        config["reconstruction"]["sparse"]
    )

    if not sparse_dir.is_absolute():
        sparse_dir = (
            Path.cwd() / sparse_dir
        )

    manifest_path = (
        output_dir / "pipeline_manifest.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconstruction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("CONFIGURATION")
    print("-" * 70)
    print("Input:", input_dir)
    print("Output:", output_dir)
    print("Frames:", frames_dir)
    print("Selected:", selected_dir)
    print("Features:", features_dir)
    print("Matches:", matches_dir)
    print("Reconstruction:", reconstruction_dir)
    print("Database:", reconstruction_database)
    print("Sparse:", sparse_dir)

    # ------------------------------------------------------------------
    # GPU CHECK
    # ------------------------------------------------------------------

    print()
    print("GPU")
    print("-" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "SuperPoint, LightGlue and Gaussian stages require CUDA."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # ------------------------------------------------------------------
    # INPUT VIDEO
    # ------------------------------------------------------------------

    video_path = load_requested_video(
        args.video,
        input_dir,
    )

    print()
    print("INPUT VIDEO")
    print("-" * 70)
    print("Video:", video_path)

    manifest = PipelineManifest(
        manifest_path
    )

    # ------------------------------------------------------------------
    # STAGE 1 — VIDEO SAMPLING
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 1 — VIDEO SAMPLING")
    print("=" * 70)

    sampler = VideoSampler(
        video_path,
        frames_dir,
        config,
    )

    frame_records = sampler.run()

    if not frame_records:
        raise RuntimeError(
            "Video sampling produced zero frames."
        )

    for record in frame_records:
        manifest.add_frame(record)

    manifest.save()

    print(
        "Frames sampled:",
        len(frame_records),
    )

    # ------------------------------------------------------------------
    # STAGE 2 — FRAME QUALITY FILTERING
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 2 — FRAME QUALITY FILTERING")
    print("=" * 70)

    frame_filter = FrameFilter(
        frame_records,
        selected_dir,
        config,
    )

    filtered_records = frame_filter.run()

    if not filtered_records:
        raise RuntimeError(
            "Frame filtering removed every frame."
        )

    print(
        "Frames after filtering:",
        len(filtered_records),
    )

    # ------------------------------------------------------------------
    # STAGE 3 — KEYFRAME SELECTION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 3 — KEYFRAME SELECTION")
    print("=" * 70)

    selector = KeyframeSelector(
        filtered_records,
        selected_dir / "keyframes",
        config,
    )

    keyframe_records = selector.run()

    if len(keyframe_records) < 3:
        raise RuntimeError(
            "Fewer than 3 keyframes were selected. "
            "Sparse reconstruction requires more views."
        )

    for record in keyframe_records:
        manifest.add_keyframe(record)

    manifest.save()

    print(
        "Keyframes:",
        len(keyframe_records),
    )

    # ------------------------------------------------------------------
    # STAGE 4 — SUPERPOINT FEATURE EXTRACTION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 4 — SUPERPOINT FEATURE EXTRACTION")
    print("=" * 70)

    feature_pipeline = FeatureExtractionPipeline(
        keyframe_records,
        features_dir,
        config,
    )

    feature_records = feature_pipeline.run()

    if len(feature_records) < 3:
        raise RuntimeError(
            "Feature extraction produced fewer than 3 feature records."
        )

    for record in feature_records:
        manifest.add_feature(record)

    manifest.save()

    print(
        "Feature files:",
        len(feature_records),
    )

    # ------------------------------------------------------------------
    # CHECK FEATURE FILES
    # ------------------------------------------------------------------

    print()
    print("FEATURE VALIDATION")
    print("-" * 70)

    missing_features = []

    for record in feature_records:
        feature_path = Path(
            record["feature_path"]
        )

        if not feature_path.exists():
            missing_features.append(
                str(feature_path)
            )

    if missing_features:
        raise RuntimeError(
            "Missing feature files:\n"
            + "\n".join(missing_features)
        )

    print("Feature files: OK")

    # ------------------------------------------------------------------
    # STAGE 5 — CAMERA
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 5 — CAMERA INITIALIZATION")
    print("=" * 70)

    first_record = feature_records[0]

    width = int(
        first_record["width"]
    )

    height = int(
        first_record["height"]
    )

    camera = CameraBuilder(
        config
    ).create(
        width,
        height,
    )

    print(
        "Camera model:",
        config["camera"]["model"],
    )

    print(
        "Resolution:",
        f"{width} x {height}",
    )

    # ------------------------------------------------------------------
    # STAGE 6 — LIGHTGLUE MATCHING + GEOMETRY + DATABASE
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 6 — MATCHING + GEOMETRIC VERIFICATION")
    print("=" * 70)

    matching_pipeline = MatchingPipeline(
        feature_records,
        features_dir,
        reconstruction_database,
        camera,
        config,
    )

    matching_result = matching_pipeline.run()

    if matching_result["pairs_tested"] == 0:
        raise RuntimeError(
            "No image pairs were generated."
        )

    if matching_result["pairs_verified"] == 0:
        raise RuntimeError(
            "No image pairs passed geometric verification."
        )

    print(
        "Pairs tested:",
        matching_result["pairs_tested"],
    )

    print(
        "Pairs verified:",
        matching_result["pairs_verified"],
    )

    print(
        "Database:",
        matching_result["database"],
    )

    for match in matching_result["matches"]:
        manifest.add_match(
            {
                "frame_a": match["frame_a"],
                "frame_b": match["frame_b"],
                "inlier_count": match["inlier_count"],
            }
        )

    manifest.save()

    # ------------------------------------------------------------------
    # DATABASE VALIDATION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DATABASE VALIDATION")
    print("=" * 70)

    if not reconstruction_database.exists():
        raise RuntimeError(
            "Matching completed but the COLMAP database "
            "was not created."
        )

    print(
        "Database exists:",
        reconstruction_database,
    )

    # ------------------------------------------------------------------
    # STAGE 7 — GLOMAP / GLOBAL RECONSTRUCTION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 7 — GLOBAL SPARSE RECONSTRUCTION")
    print("=" * 70)

    reconstruction_runner = ReconstructionRunner(
        reconstruction_database,
        selected_dir / "keyframes",
        sparse_dir,
        config,
    )

    reconstruction_result = (
        reconstruction_runner.run()
    )

    print(
        "Sparse reconstruction complete."
    )

    selected_model = reconstruction_result[
        "selected_model"
    ]

    sparse_model_path = Path(
        selected_model["path"]
    )

    print(
        "Selected model:",
        sparse_model_path,
    )

    print(
        "Registered images:",
        selected_model["images"],
    )

    print(
        "3D points:",
        selected_model["points3D"],
    )

    manifest.set_reconstruction(
        reconstruction_result
    )

    manifest.save()

    # ------------------------------------------------------------------
    # STAGE 8 — GAUSSIAN INITIALIZATION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 8 — GAUSSIAN INITIALIZATION")
    print("=" * 70)

    gaussian_dir = (
        reconstruction_dir / "gaussian"
    )

    gaussian_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialization_path = (
        gaussian_dir
        / "initialization.npz"
    )

    initializer = GaussianInitializer(
        config
    )

    initialization_result = (
        initializer.initialize(
            sparse_model_path,
            initialization_path,
        )
    )

    print(
        "Initialized Gaussians:",
        initialization_result["points"],
    )

    print(
        "Initialization:",
        initialization_path,
    )

    # ------------------------------------------------------------------
    # OPTIONAL GAUSSIAN TRAINING
    # ------------------------------------------------------------------

    training_result = None

    if args.train:
        print()
        print("=" * 70)
        print("STAGE 9 — GAUSSIAN TRAINING")
        print("=" * 70)

        from src.gaussian.pipeline import (
            GaussianPipeline,
        )

        checkpoint_path = (
            gaussian_dir
            / "checkpoint.pt"
        )

        gaussian_pipeline = GaussianPipeline(
            sparse_model_path,
            selected_dir / "keyframes",
            initialization_path,
            checkpoint_path,
            config,
        )

        training_result = (
            gaussian_pipeline.run()
        )

        print(
            "Gaussian training complete."
        )

    else:
        print()
        print("=" * 70)
        print("GAUSSIAN TRAINING SKIPPED")
        print("=" * 70)
        print(
            "Run with --train after the pipeline "
            "validation succeeds."
        )

    # ------------------------------------------------------------------
    # STAGE 10 — TRAINED GAUSSIAN RENDERING
    # ------------------------------------------------------------------

    rendering_result = None
    checkpoint_path = gaussian_dir / "checkpoint.pt"

    if checkpoint_path.exists():
        print()
        print("=" * 70)
        print("STAGE 10 — TRAINED GAUSSIAN RENDERING")
        print("=" * 70)

        rendered_dir = output_dir / "rendered"
        rendered_dir.mkdir(parents=True, exist_ok=True)

        render_command = [
            sys.executable,
            "-m",
            "src.gaussian.render_trained",
            "--config",
            str(Path("config.yaml").resolve()),
            "--sparse",
            str(sparse_model_path.resolve()),
            "--images",
            str(selected_dir.resolve()),
            "--checkpoint",
            str(checkpoint_path.resolve()),
            "--output",
            str(rendered_dir.resolve()),
        ]

        print("Rendering trained Gaussian model...")
        print("Output:", rendered_dir)

        completed = subprocess.run(
            render_command,
            cwd=str(Path.cwd()),
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Stage 10 Gaussian rendering failed. "
                f"Exit code: {completed.returncode}"
            )

        rendered_files = sorted(
            path for path in rendered_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        rendering_result = {
            "output": str(rendered_dir),
            "frames_rendered": len(rendered_files),
            "files": [str(path) for path in rendered_files],
        }

        print("Rendered frames:", len(rendered_files))
        print("Rendering complete.")
    else:
        print()
        print("=" * 70)
        print("STAGE 10 — TRAINED GAUSSIAN RENDERING SKIPPED")
        print("=" * 70)
        print("No checkpoint.pt found. Run with --train first.")

    # ------------------------------------------------------------------
    # STAGE 11 — GAUSSIAN PLY EXPORT
    # ------------------------------------------------------------------

    ply_result = None
    gaussian_ply_path = output_dir / "gaussian_model.ply"

    if checkpoint_path.exists():
        print()
        print("=" * 70)
        print("STAGE 11 — GAUSSIAN PLY EXPORT")
        print("=" * 70)

        ply_result = export_gaussian_ply(
            checkpoint_path,
            gaussian_ply_path,
        )

        print("Gaussian PLY:", gaussian_ply_path)
        print("Vertices:", ply_result["vertices"])
        print("File size:", f'{ply_result["size_mb"]:.3f} MB')
        print("PLY export complete.")
    else:
        print()
        print("STAGE 11 — GAUSSIAN PLY EXPORT SKIPPED")
        print("No trained Gaussian checkpoint is available.")

    # ------------------------------------------------------------------
    # STAGE 12 — OPEN3D INTERACTIVE 3D VIEWER
    # ------------------------------------------------------------------

    viewer_result = None

    if gaussian_ply_path.exists():
        print()
        print("=" * 70)
        print("STAGE 12 — INTERACTIVE 3D VIEWER")
        print("=" * 70)

        print("PLY:", gaussian_ply_path)
        print("Sparse model:", sparse_model_path)
        print("Starting Open3D viewer...")

        viewer_command = [
            sys.executable,
            "-m",
            "src.gaussian.viewer",
            "--input",
            str(gaussian_ply_path.resolve()),
            "--sparse",
            str(sparse_model_path.resolve()),
        ]

        # Locate the project viewer without assuming a single filename.
        viewer_candidates = [
            Path("src/gaussian/viewer.py"),
            Path("src/gaussian/open3d_viewer.py"),
            Path("viewer.py"),
        ]

        viewer_module = next(
            (path for path in viewer_candidates if path.exists()),
            None,
        )

        if viewer_module is None:
            for path in Path("src").rglob("*.py") if Path("src").exists() else []:
                try:
                    source = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "class SIH26158Viewer" in source:
                    viewer_module = path
                    break

        if viewer_module is None:
            raise FileNotFoundError(
                "Stage 12 could not locate the SIH26158 Open3D viewer. "
                "Expected viewer.py under src/gaussian or project root."
            )

        viewer_command = [
            sys.executable,
            str(viewer_module),
            "--input",
            str(gaussian_ply_path.resolve()),
            "--sparse",
            str(sparse_model_path.resolve()),
        ]

        completed = subprocess.run(
            viewer_command,
            cwd=str(Path.cwd()),
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Open3D viewer exited with code "
                f"{completed.returncode}."
            )

        viewer_result = {
            "status": "closed",
            "input": str(gaussian_ply_path),
        }

        print("3D viewer closed.")
    else:
        print()
        print("STAGE 12 — INTERACTIVE 3D VIEWER SKIPPED")
        print("No gaussian_model.ply is available.")

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------

    final_result = {
        "video": str(video_path),
        "frames": len(frame_records),
        "filtered_frames": len(filtered_records),
        "keyframes": len(keyframe_records),
        "features": len(feature_records),
        "pairs_tested": matching_result["pairs_tested"],
        "pairs_verified": matching_result["pairs_verified"],
        "database": str(reconstruction_database),
        "sparse_model": str(sparse_model_path),
        "registered_images": selected_model["images"],
        "points3D": selected_model["points3D"],
        "gaussian_initialization": initialization_result,
        "gaussian_training": training_result,
        "gaussian_rendering": rendering_result,
        "gaussian_ply": ply_result,
        "viewer": viewer_result,
    }

    report_path = output_dir / "pipeline_report.json"

    save_json(report_path, final_result)
    manifest.save()

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Report:", report_path)
    print("Manifest:", manifest_path)
    print("Sparse model:", sparse_model_path)
    print("Gaussian initialization:", initialization_path)

    if checkpoint_path.exists():
        print("Gaussian checkpoint:", checkpoint_path)

    if rendering_result is not None:
        print("Rendered views:", rendering_result["output"])

    if ply_result is not None:
        print("Gaussian PLY:", gaussian_ply_path)

    print("=" * 70)



if __name__ == "__main__":
    main()
