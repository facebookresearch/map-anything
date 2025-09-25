# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Helper functions for HuggingFace integration and model initialization.
"""

import json
import os


def load_hf_token():
    """Load HuggingFace access token from local file"""
    # Also try environment variable
    # see https://huggingface.co/docs/hub/spaces-overview#managing-secrets on options
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_MODEL_TOKEN")
    )
    if token:
        print("Loaded HuggingFace token from environment variable")
        return token

    print(
        "Warning: No HuggingFace token found. Model loading may fail for private repositories."
    )
    return None


def init_hydra_config(config_path, overrides=None):
    """Initialize Hydra config"""
    import hydra

    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)
    return cfg


def initialize_mapanything_model(high_level_config, device):
    """
    Initialize MapAnything model with three-tier fallback approach:
    1. Try HuggingFace from_pretrained()
    2. Download HF config + use local model factory + load HF weights
    3. Pure local configuration fallback

    Args:
        high_level_config (dict): Configuration dictionary containing model settings
        device (torch.device): Device to load the model on

    Returns:
        torch.nn.Module: Initialized MapAnything model
    """
    import torch
    from huggingface_hub import hf_hub_download

    from mapanything.models import init_model, MapAnything

    print("Initializing MapAnything model...")

    # Initialize Hydra config and create model from configuration
    cfg = init_hydra_config(
        high_level_config["path"], overrides=high_level_config["config_overrides"]
    )

    # Try using from_pretrained first
    try:
        print("Loading MapAnything model from_pretrained...")
        model = MapAnything.from_pretrained(high_level_config["hf_model_name"]).to(
            device
        )
        print("Loading MapAnything model from_pretrained succeeded...")
        return model
    except Exception as e:
        print(f"from_pretrained failed: {e}")
        print("Falling back to local configuration approach using hf_hub_download...")

        # Create model from local configuration instead of using from_pretrained
        # Try to download and use the config from HuggingFace Hub
        try:
            print("Downloading model configuration from HuggingFace Hub...")
            config_path = hf_hub_download(
                repo_id=high_level_config["hf_model_name"],
                filename=high_level_config["config_name"],
                token=load_hf_token(),
            )

            # Load the config from the downloaded file
            with open(config_path, "r") as f:
                downloaded_config = json.load(f)

            print("Using downloaded configuration for model initialization")
            model = init_model(
                model_str=downloaded_config.get(
                    "model_str", high_level_config["model_str"]
                ),
                model_config=downloaded_config.get(
                    "model_config", cfg.model.model_config
                ),
                torch_hub_force_reload=high_level_config.get(
                    "torch_hub_force_reload", False
                ),
            )
        except Exception as config_e:
            print(f"Failed to download/use HuggingFace config: {config_e}")
            print("Falling back to local configuration...")
            # Fall back to local configuration as before
            model = init_model(
                model_str=cfg.model.model_str,
                model_config=cfg.model.model_config,
                torch_hub_force_reload=high_level_config.get(
                    "torch_hub_force_reload", False
                ),
            )

        # Load the pretrained weights from HuggingFace Hub
        try:
            # First, let's see what files are available in the repository
            try:
                checkpoint_filename = high_level_config["checkpoint_name"]
                # Download the model weights
                checkpoint_path = hf_hub_download(
                    repo_id=high_level_config["hf_model_name"],
                    filename=checkpoint_filename,
                    token=load_hf_token(),
                )

                # Load the weights
                print("start loading checkpoint")
                if checkpoint_filename.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    checkpoint = load_file(checkpoint_path)
                else:
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )

                print("start loading state_dict")
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"], strict=False)
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)

                print(
                    f"Successfully loaded pretrained weights from HuggingFace Hub ({checkpoint_filename})"
                )

            except Exception as inner_e:
                print(f"Error listing repository files or loading weights: {inner_e}")
                raise inner_e

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Proceeding with randomly initialized model...")

        model = model.to(device)
        return model
