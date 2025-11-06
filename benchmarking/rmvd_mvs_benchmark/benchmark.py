# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
RMVD Benchmark Executable for MapAnything

This script serves as the main entry point for Robust-MVD benchmarking models in the MapAnything project.
It uses Hydra for configuration management and redirects all output to logging.

Usage:
    python train.py [hydra_options]
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from adaptors import RMVD_MAPA_Wrapper
from omegaconf import DictConfig, OmegaConf
from rmvd import create_dataset, create_evaluation

from morphcloud.models import init_model
from morphcloud.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


@torch.inference_mode()
def run_benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Initialize RMVD test dataset
    inference_resolution_wh = eval(args.evaluation_resolution)

    root_dir = os.path.join(
        args.external_benchmark_data_root_data_dir, args.eval_dataset
    )
    print(f"Using data root dir: {root_dir}")
    assert os.path.exists(root_dir), f"Data root dir {root_dir} does not exist!"

    dataset = create_dataset(
        args.eval_dataset,
        "mvd",
        input_size=tuple(inference_resolution_wh[::-1]),
        root=root_dir,
    )

    # Initialize RMVD evaluation
    additional_info = []
    if "intrinsics" in args.evaluation_conditioning:
        additional_info.append("intrinsics")
    if "pose" in args.evaluation_conditioning:
        additional_info.append("poses")

    # Run evaluation
    evaluation = create_evaluation(
        evaluation_type="mvd",
        out_dir=args.output_dir,
        inputs=additional_info,
        alignment=(
            None if args.evaluation_alignment == "none" else args.evaluation_alignment
        ),
        eval_uncertainty=False,
        max_source_views=7,  # Following MVSAnywhere https://github.com/nianticlabs/mvsanywhere/blob/1a705b48281aa99fe5d9dd4b29b5f2d43e32b801/src/mvsanywhere/test_rmvd.py#L38C9-L38C28
    )

    # Load Model
    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)  # Move model to device

    # Load pretrained model
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    # Wrap the pretrained model with RMVD Wrapper to work with its benchmark
    wrapped_model = RMVD_MAPA_Wrapper(
        name=args.model.model_str,
        model=model,
        data_norm_type=args.model.data_norm_type,
        use_amp=bool(args.amp),
        amp_dtype=args.amp_dtype,
        inference_conditioning=args.evaluation_conditioning,
        evaluate_single_view=(args.evaluation_views == "single_view"),
    )

    # Run the evaluation
    evaluation(dataset=dataset, model=wrapped_model)

    # Dump the evaluation setting into a text file for double confirmation
    setting_file = os.path.join(args.output_dir, "setting.json")
    setting = {
        "dataset": args.eval_dataset,
        "resolution_wh": inference_resolution_wh,
        "conditioning": args.evaluation_conditioning,
        "alignment": args.evaluation_alignment,
        "num_views": args.evaluation_views,
    }

    with open(setting_file, "w") as f:
        json.dump(setting, f, indent=4)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rmvd_benchmark"
)
def launch_benchmark(cfg: DictConfig):
    """
    Execute the training process with the provided configuration.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra
    """
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the training
    run_benchmark(cfg)


if __name__ == "__main__":
    launch_benchmark()
