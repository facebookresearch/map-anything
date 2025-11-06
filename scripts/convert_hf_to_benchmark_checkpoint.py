# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Convert HuggingFace MapAnything checkpoint to benchmark-compatible format.

This script loads a HuggingFace model and saves it in the format expected
by the benchmarking system, which requires a dictionary containing 'model' (state_dict) key.

Usage:
    python scripts/convert_hf_to_benchmark_checkpoint.py \
        --hf_model_name "facebook/map-anything" \
        --output_path /path/to/output/checkpoint.pth \
"""

import argparse
import sys
from pathlib import Path

import torch

from morphcloud.models import MapAnything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace MapAnything checkpoint to benchmark format"
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="facebook/map-anything",
        help="HuggingFace model name (default: facebook/map-anything)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the converted checkpoint (.pth file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for loading (auto, cpu, cuda) (default: auto)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)
    print(f"Using device: {device}")

    # Use Apache model if requested
    hf_model_name = args.hf_model_name
    if args.apache:
        hf_model_name = "facebook/map-anything-apache"

    print(f"Loading HuggingFace model: {hf_model_name}")
    model = MapAnything.from_pretrained(hf_model_name).to(device)
    print("Successfully loaded model using MapAnything.from_pretrained()")

    # Get model state dict
    print("Extracting model state dict...")
    model_state_dict = model.state_dict()

    # Create checkpoint dictionary in the expected format
    checkpoint = {
        "model": model_state_dict,
        "hf_model_name": hf_model_name,
        "conversion_info": {
            "original_format": "huggingface",
            "converted_by": "convert_hf_to_benchmark_checkpoint.py",
            "device_used": str(device),
        },
    }

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    print(f"Saving checkpoint to: {args.output_path}")
    torch.save(checkpoint, args.output_path)

    # Verify the saved checkpoint
    print("Verifying saved checkpoint...")
    try:
        loaded_checkpoint = torch.load(
            args.output_path, map_location="cpu", weights_only=False
        )
        print("✓ Checkpoint saved successfully")
        print(f"✓ Contains 'model' key: {'model' in loaded_checkpoint}")

        if "hf_model_name" in loaded_checkpoint:
            print(f"✓ Original HF model: {loaded_checkpoint['hf_model_name']}")
    except Exception as e:
        print(f"✗ Error verifying checkpoint: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Conversion completed successfully!")
    print(f"HuggingFace model: {hf_model_name}")
    print(f"Output checkpoint: {args.output_path}")
    print("\nTo use this checkpoint in benchmarking, update your config:")
    print(f'model.pretrained: "{args.output_path}"')
    print("=" * 50)


if __name__ == "__main__":
    main()
