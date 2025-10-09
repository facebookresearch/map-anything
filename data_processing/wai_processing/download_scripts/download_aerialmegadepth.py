# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Download AerialMegaDepth dataset from HuggingFace.

References: https://github.com/kvuong2711/aerial-megadepth/blob/main/data_generation/download_data_hf.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from wai_processing.utils.download import (
    extract_zip_archives,
)


# Configuration for AerialMegaDepth dataset
REPO_ID = "kvuong2711/aerialmegadepth"
ALLOW_PATTERNS = ("**.zip", "aerial_megadepth_all.npz")
DEFAULT_MAX_WORKERS = 8
ZIP_DIR_NAME = "aerialmegadepth_zip"
EXTRACT_DIR_NAME = "aerialmegadepth"


def download_archives(zip_dir: Path, max_workers: int):
    """Download dataset archives into ``zip_dir``."""
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} archives to {zip_dir}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(zip_dir),
        max_workers=max_workers,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    print(f"Download complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Aerial MegaDepth dataset and optionally extract archives.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Base directory for downloaded archives and extracted data.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of parallel workers used by the Hugging Face downloader.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    zip_dir = target_dir / ZIP_DIR_NAME
    extract_dir = target_dir / EXTRACT_DIR_NAME

    # 1. Download zip files from huggingface
    download_archives(zip_dir, max_workers=args.max_workers)

    # 2. Extract zip files
    extract_zip_archives(target_dir=zip_dir, output_dir=extract_dir, n_workers=args.max_workers)

    # 3. Move the aerial_megadepth_all.npz to the extract_dir
    shutil.move(zip_dir / "aerial_megadepth_all.npz", extract_dir / "aerial_megadepth_all.npz")

    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()