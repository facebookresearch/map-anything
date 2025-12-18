# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Model Factory for MapAnything
"""

import importlib.util
import logging
import warnings

import numpy as np
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


__all__ = ["model_factory", "get_available_models"]
