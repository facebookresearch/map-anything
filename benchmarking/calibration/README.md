# Single-View Image Calibration Benchmark

## Overview

This benchmark evaluates a method's ability to predict camera intrinsics from single images without any geometric inputs. The model must recover camera ray directions (equivalent to intrinsic parameters) from RGB images alone across multiple datasets (test splits of ETH3D, ScanNet++V2, and TartanAirV2-WB datasets in WAI format) and varying aspect ratios.

## Prepare Evaluation Data

### Option 1: Download Pre-processed Data (Recommended)

Download and extract the pre-processed WAI format benchmarking data directly from HuggingFace:

```bash
python data_processing/download_and_extract_benchmarking_data.py \
    --download --extract \
    --output_dir "<your_data_dir>/map-anything-benchmarking-dataset"
```

The downloaded data includes test splits for ETH3D, ScanNet++V2, and TartanAirV2-WB in WAI format. Run with `--help` for additional options.

### Option 2: Process Data Yourself

Alternatively, process the test datasets to WAI format manually. See [Data Processing README](../../data_processing/README.md) for WAI format details and conversion instructions.

## Test Processed Data (Optional)

Verify your processed data by running the main calls of the dataloaders (use `--viz` option for Rerun visualization). For example, to visualize the dataloader outputs for ETH3D:

```bash
python mapanything/datasets/wai/eth3d.py --viz
```

See the main call in each dataloader file for usage details.

## Prepare MapAnything Checkpoint

The benchmarking system expects trained checkpoints in a specific format with a `model` (state_dict) key. Convert HuggingFace models to the required benchmark format:

```bash
# Convert default CC-BY-NC model
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --output_path checkpoints/facebook_map-anything.pth

# Convert Apache 2.0 model for commercial use
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

## Run Benchmark

All the original benchmarking bash scripts are available at:
- `bash_scripts/benchmark/calibration/`

Update the machine configuration (your machine in `configs/machine/`) and model checkpoint paths in the respective bash scripts, then execute:

```bash
bash bash_scripts/benchmark/calibration/mapanything.sh
```

Results will be saved to the configured output directory.
