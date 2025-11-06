# RobustMVD Benchmark

## RobustMVD Installation

If you have not installed with the "[all]" option, run the following command in the base directory of this repo:

```bash
pip install -e ".[rmvd]"
```

This will install the RobustMVD library for the benchmark.

## Prepare Evaluation Data

Follow [instructions by RMVD](https://github.com/lmb-freiburg/robustmvd/tree/master/rmvd/data) to prepare the datasets KITTI and Scannet. Place all the data under the `external_benchmark_data_root_data_dir` as specified in the machine config (for e.g., see any one of the `configs/machine/*.yaml` files).

## Prepare MapAnything Checkpoint

The benchmarking system expects trained checkpoints in a specific format with a `model` (state_dict) key. Convert HuggingFace models to the required format:

```bash
# Convert default CC-BY-NC model
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --output_path checkpoints/facebook_map-anything.pth

# Convert Apache 2.0 model for commercial use
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

## Generate Benchmark Scripts and Run the Benchmark

1. **Update machine configuration**: Modify the `machine` variable in `bash_scripts/benchmark/rmvd_mvs_benchmark/generate_benchmark_scripts.py` to match your machine config name in `configs/machine/`.

2. **Update model checkpoint path**: In the same file, update the `model.pretrained` entry in the `get_model_settings()` function to point to your converted checkpoint:

   ```python
   def get_model_settings(model: str, dataset: str):
       if model == "morphcloud":
           return {
               "model": "morphcloud",
               "model.pretrained": "/path/to/your/converted/checkpoint.pth",  # Update this path (can be any checkpoint from training or the above generated checkpoints)
               "evaluation_resolution": "\\${dataset.resolution_options.518_1_33_ar}"
               if dataset != "kitti"
               else "\\${dataset.resolution_options.518_3_20_ar}",
           }
   ```

3. **Generate benchmark scripts**: Run the script to generate all shell files:
   ```bash
   python bash_scripts/benchmark/rmvd_mvs_benchmark/generate_benchmark_scripts.py
   ```

4. **Run benchmarks**: Execute the generated shell scripts. Each shell file corresponds to one number in the RobustMVD table of the paper. You will find the output in `outputs/morphcloud/benchmarking`.
