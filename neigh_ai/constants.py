import os
from pathlib import Path


def get_images_folder() -> Path:
    env = os.environ.get("ENV", "prod")
    if env == "test":
        return Path(__file__).parent.parent / "tests" / "assets" / "images"

    else:
        return Path(__file__).parent.parent / "data" / "images"


def get_hip_sheet_path() -> Path:
    env = os.environ.get("ENV", "prod")
    if env == "test":
        return Path(__file__).parent.parent / "tests" / "assets" / "hip_sheet.csv"

    else:
        return Path(__file__).parent.parent / "data" / "hip_sheet.csv"
