# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Executable for MapAnything

This script serves as the main entry point for training models in the MapAnything project.
It uses Hydra for configuration management and redirects all output to logging.

Usage:
    python train.py [hydra_options]
"""

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from mapanything.train.training import train
from mapanything.train.training_jepa import train_jepa
from mapanything.train.training_probe import train_probe
from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def execute_training(cfg: DictConfig):
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
    if cfg.train_params.jepa:
        train_jepa(cfg)
    elif cfg.train_params.probe:
        train_probe(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    execute_training()
