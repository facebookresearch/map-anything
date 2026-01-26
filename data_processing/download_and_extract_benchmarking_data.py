# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# Download and extract MapAnything benchmarking dataset from HuggingFace Hub
# --------------------------------------------------------
import zipfile
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

# HuggingFace repository for benchmarking data
HF_REPO_ID = "facebook/map-anything-benchmarking"


def extract_zip_file(
    zip_path: Path, extract_dir: Path, delete_after: bool
) -> tuple[Path, str | None]:
    """Extract a single zip file.

    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
        delete_after: Whether to delete the zip file after successful extraction

    Returns:
        Tuple of (zip_path, error_message). error_message is None on success.
    """
    try:
        # Determine the extraction target directory
        # Maintain the same relative structure as the zip file
        relative_path = zip_path.relative_to(zip_path.parent.parent)
        target_dir = extract_dir / relative_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        if delete_after:
            zip_path.unlink()

        return zip_path, None
    except Exception as e:
        return zip_path, str(e)


def extract_all_zips(
    download_dir: Path, extract_dir: Path, delete_zips: bool, num_workers: int = 24
):
    """Extract all zip files in the download directory.

    Args:
        download_dir: Directory containing downloaded zip files
        extract_dir: Directory to extract files to
        delete_zips: Whether to delete zip files after successful extraction
        num_workers: Number of parallel workers for extraction
    """
    # Find all zip files
    zip_files = list(download_dir.rglob("*.zip"))

    if not zip_files:
        print("No zip files found to extract")
        return

    print(f"Found {len(zip_files)} zip files to extract")

    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(extract_zip_file, zf, extract_dir, delete_zips): zf
            for zf in zip_files
        }
        for future in tqdm(
            as_completed(futures), total=len(zip_files), desc="Extracting"
        ):
            result_path, error = future.result()
            if error:
                errors.append((result_path, error))

    if errors:
        print(f"\nFailed to extract {len(errors)} zip file(s):")
        for path, error in errors:
            print(f"  - {path}: {error}")
    else:
        print("Successfully extracted all zip files")
        if delete_zips:
            print("Zip files have been deleted")


def download_dataset(output_dir: str, num_workers: int = 24) -> Path:
    """Download the benchmarking dataset from HuggingFace Hub.

    Args:
        output_dir: Target directory for the downloaded data
        num_workers: Number of parallel workers for download

    Returns:
        Path to the downloaded dataset
    """
    print(f"Downloading dataset from HuggingFace: {HF_REPO_ID}")
    print(f"Target directory: {output_dir}")
    print(f"Using {num_workers} parallel workers")

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=output_dir,
        max_workers=num_workers,
    )

    print(f"Successfully downloaded dataset to: {local_dir}")
    return Path(local_dir)


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and extract MapAnything benchmarking dataset from HuggingFace Hub"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from HuggingFace",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract zip files after download",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete zip files after successful extraction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target directory for downloaded data",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default=None,
        help="Target directory for extracted data (default: same as output_dir)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="Number of parallel workers for download/extraction",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Check if any action is requested
    if not args.download and not args.extract:
        print("No action specified. Use --download and/or --extract flags.")
        print("Run with --help for usage information.")
        exit(0)

    output_dir = Path(args.output_dir)
    extract_dir = Path(args.extract_dir) if args.extract_dir else output_dir

    # Download dataset if requested
    if args.download:
        download_dataset(str(output_dir), args.num_workers)

    # Extract zip files if requested
    if args.extract:
        if not output_dir.exists():
            print(f"Error: Output directory does not exist: {output_dir}")
            print("Please download the dataset first with --download flag.")
            exit(1)
        extract_all_zips(output_dir, extract_dir, args.delete_zips, args.num_workers)
