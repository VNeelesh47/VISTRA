from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    config_path = ROOT_DIR / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["paths"] = {
        "root": ROOT_DIR,
        "input": ROOT_DIR / config["input"]["video_dir"],
        "output": ROOT_DIR / config["output"]["root"],
        "frames": ROOT_DIR / config["output"]["frames"],
        "selected": ROOT_DIR / config["output"]["selected"],
        "features": ROOT_DIR / config["output"]["features"],
        "matches": ROOT_DIR / config["output"]["matches"],
        "reconstruction": ROOT_DIR / config["reconstruction"]["root"],
    }

    config["tools"] = {
        "colmap": os.getenv("COLMAP_PATH")
    }

    return config