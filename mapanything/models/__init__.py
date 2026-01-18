# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Model Factory for MapAnything
"""

import importlib.util
import logging
import os
import warnings
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Core models that are always available
from mapanything.models.mapanything import (
    MapAnything,
    MapAnythingAblations,
    ModularDUSt3R,
)

# Suppress DINOv2 warnings
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="xFormers is not available", category=UserWarning
)


def resolve_special_float(value):
    if value == "inf":
        return np.inf
    elif value == "-inf":
        return -np.inf
    else:
        raise ValueError(f"Unknown special float value: {value}")


def init_model(
    model_str: str, model_config: DictConfig, torch_hub_force_reload: bool = False
):
    """
    Initialize a model using OmegaConf configuration.

    Args:
        model_str (str): Name of the model class to create.
        model_config (DictConfig): OmegaConf model configuration.
        torch_hub_force_reload (bool): Whether to force reload relevant parts of the model from torch hub.
    """
    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)
    model_dict = OmegaConf.to_container(model_config, resolve=True)
    model = model_factory(
        model_str, torch_hub_force_reload=torch_hub_force_reload, **model_dict
    )

    return model


def _init_hydra_config(config_path: str, overrides: Optional[List[str]] = None):
    """
    Initialize Hydra config with proper composition and interpolation resolution.

    Args:
        config_path: Relative path to the config file (e.g., "configs/train.yaml")
        overrides: Optional list of Hydra overrides (e.g., ["model=vggt"])

    Returns:
        Composed OmegaConf config with all interpolations resolved
    """
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]

    # Get the project root (parent of mapanything package)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(package_dir))
    abs_config_dir = os.path.join(project_root, config_dir)

    # Get relative path from this module to the config directory
    relative_path = os.path.relpath(abs_config_dir, package_dir)

    # Clear any existing Hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    hydra.initialize(version_base=None, config_path=relative_path)

    # Compose the config with overrides
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg


def init_model_from_config(
    model_name: str,
    device: str = "cuda",
    machine: str = "default",
) -> torch.nn.Module:
    """
    Initialize a model using Hydra config composition.

    Models automatically load their pretrained weights via their config.

    Args:
        model_name: Name of the model config (e.g., "vggt", "pi3x", "mapanything")
        device: Device to load model on (default: "cuda")

    Returns:
        Initialized model on the specified device in eval mode

    Example:
        >>> from mapanything.models import init_model_from_config
        >>> model = init_model_from_config("vggt", device="cuda")
    """
    # Use train.yaml as base config with model override for proper Hydra composition
    config_path = "configs/train.yaml"
    overrides = [f"model={model_name}", f"machine={machine}"]

    # Use Hydra to properly compose and resolve the config
    config = _init_hydra_config(config_path, overrides=overrides)

    # Initialize model using the factory
    model = init_model(
        model_str=config.model.model_str,
        model_config=config.model.model_config,
        torch_hub_force_reload=False,
    )

    model = model.to(device)

    return model


# Define model configurations with import paths
MODEL_CONFIGS = {
    # Core models
    "mapanything": {
        "class": MapAnything,
    },
    "mapanything_ablations": {
        "class": MapAnythingAblations,
    },
    "modular_dust3r": {
        "class": ModularDUSt3R,
    },
    # External models
    "anycalib": {
        "module": "mapanything.models.external.anycalib",
        "class_name": "AnyCalibWrapper",
    },
    "da3": {
        "module": "mapanything.models.external.da3",
        "class_name": "DA3Wrapper",
    },
    "dust3r": {
        "module": "mapanything.models.external.dust3r",
        "class_name": "DUSt3RBAWrapper",
    },
    "mast3r": {
        "module": "mapanything.models.external.mast3r",
        "class_name": "MASt3RSGAWrapper",
    },
    "moge": {
        "module": "mapanything.models.external.moge",
        "class_name": "MoGeWrapper",
    },
    "must3r": {
        "module": "mapanything.models.external.must3r",
        "class_name": "MUSt3RWrapper",
    },
    "pi3": {
        "module": "mapanything.models.external.pi3",
        "class_name": "Pi3Wrapper",
    },
    "pi3x": {
        "module": "mapanything.models.external.pi3x",
        "class_name": "Pi3XWrapper",
    },
    "pow3r": {
        "module": "mapanything.models.external.pow3r",
        "class_name": "Pow3RWrapper",
    },
    "pow3r_ba": {
        "module": "mapanything.models.external.pow3r",
        "class_name": "Pow3RBAWrapper",
    },
    "vggt": {
        "module": "mapanything.models.external.vggt",
        "class_name": "VGGTWrapper",
    },
    # Add other model classes here
}


def check_module_exists(module_path):
    """
    Check if a module can be imported without actually importing it.

    Args:
        module_path (str): The path to the module to check.

    Returns:
        bool: True if the module can be imported, False otherwise.
    """
    return importlib.util.find_spec(module_path) is not None


def model_factory(model_str: str, **kwargs):
    """
    Model factory for MapAnything.

    Args:
        model_str (str): Name of the model to create.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
       nn.Module: An instance of the specified model.
    """
    if model_str not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_str}. Valid options are: {', '.join(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[model_str]

    # Handle core models directly
    if "class" in model_config:
        model_class = model_config["class"]
    # Handle external models with dynamic imports
    elif "module" in model_config:
        module_path = model_config["module"]
        class_name = model_config["class_name"]

        # Check if the module can be imported
        if not check_module_exists(module_path):
            raise ImportError(
                f"Model '{model_str}' requires module '{module_path}' which is not installed. "
                f"Please install the corresponding submodule or package."
            )

        # Dynamically import the module and get the class
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_path}: {str(e)}"
            )
    else:
        raise ValueError(f"Invalid model configuration for {model_str}")

    print(f"Initializing {model_class} with kwargs: {kwargs}")
    if model_str != "org_dust3r":
        return model_class(**kwargs)
    else:
        eval_str = kwargs.get("model_eval_str", None)
        return eval(eval_str)


def get_available_models() -> list:
    """
    Get a list of available models in MapAnything.

    Returns:
        list: A list of available model names.
    """
    return list(MODEL_CONFIGS.keys())


__all__ = [
    "model_factory",
    "init_model",
    "init_model_from_config",
    "get_available_models",
]
