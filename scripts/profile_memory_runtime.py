# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Profile GPU memory usage and inference speed of MapAnything and external models.

This script profiles models across increasing view counts and outputs:
- JSON file with profiling results
- Memory usage plot (Peak GPU Memory vs Number of Views)
- Speed plot (Inference Frequency Hz vs Number of Views)

Example usage:
    # Profile MapAnything only
    python scripts/profile_memory_runtime.py \
        --output_dir /path/to/results

    # Compare with external models
    python scripts/profile_memory_runtime.py \
        --output_dir /path/to/results \
        --external_models vggt pi3x must3r

    # Custom view counts
    python scripts/profile_memory_runtime.py \
        --output_dir /path/to/results \
        --num_views 2 4 8 16 32

    # Include MapAnything V1 models for comparison
    python scripts/profile_memory_runtime.py \
        --output_dir /path/to/results \
        --profile_v1
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mapanything.models import init_model_from_config, MapAnything

matplotlib.use("Agg")  # Use non-interactive backend

# ============================================================================
# Inline Plotting Utilities (copied from private/scripts/plot_mv_benchmarking_graphs.py)
# ============================================================================

# Color palette for methods
COLOR_PALETTES = {
    "brown": "tab:brown",
    "pink": "tab:pink",
    "orange": "tab:orange",
    "purple": "tab:purple",
    "blue": "tab:blue",
    "green": "tab:green",
    "red": "tab:red",
    "cyan": "tab:cyan",
    "olive": "tab:olive",
    "gray": "tab:gray",
    "teal": "#4ECDC4",
    "gold": "#FFD93D",
    "coral": "#FF6B6B",
}

# Method colors mapping
METHOD_COLORS = {
    "mapanything": COLOR_PALETTES["orange"],
    "mapanything_mem_efficient": COLOR_PALETTES["coral"],
    "mapanything_v1": COLOR_PALETTES["gray"],
    "mapanything_v1_mem_efficient": "#C0C0C0",
    "vggt": COLOR_PALETTES["purple"],
    "pi3": COLOR_PALETTES["red"],
    "pi3x": "#800000",  # Maroon
    "dust3r": COLOR_PALETTES["blue"],
    "mast3r": COLOR_PALETTES["green"],
    "must3r": COLOR_PALETTES["olive"],
    "pow3r": COLOR_PALETTES["brown"],
    "pow3r_ba": COLOR_PALETTES["brown"],
    "da3": COLOR_PALETTES["teal"],
    "da3_nested": COLOR_PALETTES["pink"],
    "moge_1": COLOR_PALETTES["cyan"],
    "moge_2": COLOR_PALETTES["cyan"],
}

# Method display names
METHOD_NAMES = {
    "mapanything": "MapAnything",
    "mapanything_mem_efficient": "MapAnything (Mem Efficient)",
    "mapanything_v1": "MapAnything V1",
    "mapanything_v1_mem_efficient": "MapAnything V1 (Mem Efficient)",
    "vggt": "VGGT",
    "pi3": r"$\pi^3$",
    "pi3x": r"$\pi^3$-X",
    "dust3r": "DUSt3R-BA",
    "mast3r": "MASt3R-SGA",
    "must3r": "MUSt3R",
    "pow3r": "Pow3R",
    "pow3r_ba": "Pow3R-BA",
    "da3": "Depth-Anything-3",
    "da3_nested": "Depth-Anything-3-Nested",
    "moge_1": "MoGe-1",
    "moge_2": "MoGe-2",
}

# Additional distinct colors for unknown methods
DISTINCT_COLORS = [
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d3",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]

# Global color assignment cache
_method_color_cache = {}


def get_method_color(method_name: str) -> str:
    """Get color for a method, assigning new colors for unknown methods."""
    global _method_color_cache

    # Check predefined colors
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]

    # Check cache
    if method_name in _method_color_cache:
        return _method_color_cache[method_name]

    # Assign new color
    used_colors = set(METHOD_COLORS.values()) | set(_method_color_cache.values())
    for color in DISTINCT_COLORS:
        if color not in used_colors:
            _method_color_cache[method_name] = color
            return color

    # Fallback
    return "#333333"


def get_method_display_name(method_name: str) -> str:
    """Get display name for a method."""
    if method_name in METHOD_NAMES:
        return METHOD_NAMES[method_name]
    return method_name.replace("_", " ").title()


def sort_methods_mapanything_first(method_names: List[str]) -> List[str]:
    """
    Sort method names so MapAnything variants appear first in legend.

    Order:
    1. mapanything (current version, default mode)
    2. mapanything_mem_efficient (current version, memory efficient)
    3. mapanything_v1 (V1, default mode)
    4. mapanything_v1_mem_efficient (V1, memory efficient)
    5. All other methods in their original order
    """
    # Define priority order for MapAnything variants (lower = higher priority)
    mapanything_priority = {
        "mapanything": 0,
        "mapanything_mem_efficient": 1,
        "mapanything_v1": 2,
        "mapanything_v1_mem_efficient": 3,
    }

    def sort_key(name: str) -> tuple:
        if name in mapanything_priority:
            return (0, mapanything_priority[name])
        # Non-MapAnything methods get priority 1, maintain original order
        return (1, method_names.index(name))

    return sorted(method_names, key=sort_key)


# ============================================================================
# Image Normalization Constants
# ============================================================================

# DINOv2 normalization (ImageNet stats)
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225])

# Model resolution and normalization type mapping
MODEL_CONFIG = {
    "mapanything": {"resolution": 518, "norm_type": "dinov2", "patch_size": 14},
    "mapanything_v1": {"resolution": 518, "norm_type": "dinov2", "patch_size": 14},
    "vggt": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
    "vggt_commercial": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
    "pi3": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
    "pi3x": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
    "dust3r": {"resolution": 512, "norm_type": "dust3r", "patch_size": 16},
    "mast3r": {"resolution": 512, "norm_type": "dust3r", "patch_size": 16},
    "must3r": {"resolution": 512, "norm_type": "dust3r", "patch_size": 16},
    "pow3r": {"resolution": 512, "norm_type": "dust3r", "patch_size": 16},
    "pow3r_ba": {"resolution": 512, "norm_type": "dust3r", "patch_size": 16},
    "da3": {"resolution": 504, "norm_type": "dinov2", "patch_size": 14},
    "da3_nested": {"resolution": 504, "norm_type": "dinov2", "patch_size": 14},
    "moge_1": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
    "moge_2": {"resolution": 518, "norm_type": "identity", "patch_size": 14},
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration, with fallback defaults."""
    if model_name in MODEL_CONFIG:
        return MODEL_CONFIG[model_name]
    # Default config
    return {"resolution": 518, "norm_type": "dinov2", "patch_size": 14}


# ============================================================================
# Model Loading
# ============================================================================


def load_model_with_metadata(
    model_name: str,
    device: str,
) -> Tuple[torch.nn.Module, str, int]:
    """
    Load a model using Hydra config composition and return with metadata.

    Uses init_model_from_config from mapanything.models package.

    Args:
        model_name: Name of the model (e.g., "vggt", "pi3x", "mapanything")
        device: Device to load model on

    Returns:
        model: Initialized model
        data_norm_type: Normalization type for images
        resolution: Model's expected resolution
    """
    # Use the centralized model loading function
    model = init_model_from_config(model_name, device=device)

    # Get model config for resolution and norm type
    model_cfg = get_model_config(model_name)
    data_norm_type = model_cfg["norm_type"]
    resolution = model_cfg["resolution"]

    return model, data_norm_type, resolution


def load_mapanything_from_pretrained(
    device: str,
    checkpoint_path: Optional[str] = None,
    use_apache: bool = False,
    use_v1: bool = False,
) -> Tuple[torch.nn.Module, str, int]:
    """
    Load MapAnything using from_pretrained or checkpoint.

    Args:
        device: Device to load model on
        checkpoint_path: Optional path to checkpoint (overrides pretrained)
        use_apache: Whether to use Apache 2.0 licensed model
        use_v1: Whether to use V1 (deprecated) models

    Returns:
        model: Initialized MapAnything model
        data_norm_type: Normalization type
        resolution: Model resolution
    """
    model_cfg = get_model_config("mapanything_v1" if use_v1 else "mapanything")

    if checkpoint_path is not None:
        # Load from checkpoint
        model = init_model_from_config("mapanything", device=device)
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        # Load from HuggingFace
        if use_v1:
            model_id = (
                "facebook/map-anything-apache-v1"
                if use_apache
                else "facebook/map-anything-v1"
            )
        else:
            model_id = (
                "facebook/map-anything-apache"
                if use_apache
                else "facebook/map-anything"
            )
        print(f"Loading MapAnything from: {model_id}")
        model = MapAnything.from_pretrained(model_id).to(device)
        model.eval()

    return model, model_cfg["norm_type"], model_cfg["resolution"]


# ============================================================================
# Synthetic Image Generation
# ============================================================================


def generate_synthetic_views(
    num_views: int,
    resolution: int,
    data_norm_type: str,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate synthetic image views for profiling.

    Args:
        num_views: Number of views to generate
        resolution: Image resolution (square images)
        data_norm_type: Normalization type ("dinov2", "dust3r", "identity")
        device: Device to create tensors on

    Returns:
        List of view dictionaries with normalized images
    """
    H, W = resolution, resolution

    views = []
    for _ in range(num_views):
        # Generate random image in [0, 1]
        img = torch.rand(1, 3, H, W, device=device)

        # Apply normalization based on type
        if data_norm_type == "dinov2":
            mean = DINOV2_MEAN.view(1, 3, 1, 1).to(device)
            std = DINOV2_STD.view(1, 3, 1, 1).to(device)
            img = (img - mean) / std
        elif data_norm_type == "dust3r":
            img = (img - 0.5) / 0.5
        # identity: no normalization needed

        views.append(
            {
                "img": img,
                "true_shape": np.array([[H, W]]),
                "idx": [0],
                "instance": ["synthetic"],
                "data_norm_type": [data_norm_type],
            }
        )

    return views


# ============================================================================
# Profiling Logic
# ============================================================================


def profile_model(
    model: torch.nn.Module,
    views: List[Dict[str, torch.Tensor]],
    num_warmup: int = 3,
    num_timed: int = 5,
    memory_efficient: bool = False,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    minibatch_size: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Profile a model's memory and runtime.

    Args:
        model: Model to profile
        views: List of input view dictionaries
        num_warmup: Number of warmup iterations
        num_timed: Number of timed iterations
        memory_efficient: Whether to use memory-efficient inference (MapAnything only)
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype ("bf16", "fp16", or "fp32")
        minibatch_size: Minibatch size for memory-efficient inference (MapAnything only)

    Returns:
        Dictionary with memory and frequency statistics
    """
    is_mapanything = isinstance(model, MapAnything)

    # Determine AMP dtype
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif amp_dtype == "fp16" or (
        amp_dtype == "bf16" and not torch.cuda.is_bf16_supported()
    ):
        dtype = torch.float16
    else:
        dtype = torch.float32

    def run_inference():
        with torch.no_grad():
            if is_mapanything:
                if use_amp:
                    with torch.autocast("cuda", dtype=dtype):
                        model(
                            views,
                            memory_efficient_inference=memory_efficient,
                            minibatch_size=minibatch_size,
                        )
                else:
                    model(
                        views,
                        memory_efficient_inference=memory_efficient,
                        minibatch_size=minibatch_size,
                    )
            else:
                if use_amp:
                    with torch.autocast("cuda", dtype=dtype):
                        model(views)
                else:
                    model(views)

    # Warmup runs
    for _ in range(num_warmup):
        run_inference()
        torch.cuda.synchronize()

    # Clear cache and reset memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    runtimes = []
    for _ in range(num_timed):
        torch.cuda.synchronize()
        start = time.perf_counter()

        run_inference()

        torch.cuda.synchronize()
        end = time.perf_counter()
        runtimes.append(end - start)

    # Get peak memory (in GB)
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024**3)

    # Calculate frequency (Hz)
    mean_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes))
    frequency_hz = 1.0 / mean_runtime if mean_runtime > 0 else 0
    frequency_std = float(np.std([1.0 / r for r in runtimes if r > 0]))

    return {
        "memory_gb": {"mean": float(peak_memory_gb), "std": 0.0},
        "frequency_hz": {"mean": frequency_hz, "std": frequency_std},
        "runtime_s": {"mean": mean_runtime, "std": std_runtime},
    }


def profile_model_across_views(
    model: torch.nn.Module,
    model_name: str,
    data_norm_type: str,
    resolution: int,
    num_views_list: List[int],
    device: str,
    num_warmup: int = 3,
    num_timed: int = 5,
    memory_efficient: bool = False,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    minibatch_size: int = 1,
    existing_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Profile a model across multiple view counts.

    Args:
        model: Model to profile
        model_name: Name of the model for logging
        data_norm_type: Image normalization type
        resolution: Image resolution
        num_views_list: List of view counts to profile
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_timed: Number of timed iterations
        memory_efficient: Whether to use memory-efficient inference
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype
        minibatch_size: Minibatch size for memory-efficient inference
        existing_results: Optional dict of existing results to resume from

    Returns:
        Dictionary mapping view counts to profiling results
    """
    # Start with existing results if provided
    results = dict(existing_results) if existing_results else {}

    for i, num_views in enumerate(tqdm(num_views_list, desc=f"Profiling {model_name}")):
        # Skip if already profiled (either successfully or OOM)
        existing = results.get(str(num_views))
        if existing is not None:
            if existing == "oom":
                print(f"  {num_views} views: Previously OOM/limit, skipping")
            else:
                print(f"  {num_views} views: Already profiled, skipping")
            continue

        try:
            # Generate synthetic views
            views = generate_synthetic_views(
                num_views, resolution, data_norm_type, device
            )

            # Profile
            result = profile_model(
                model,
                views,
                num_warmup=num_warmup,
                num_timed=num_timed,
                memory_efficient=memory_efficient,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                minibatch_size=minibatch_size,
            )
            results[str(num_views)] = result

            print(
                f"  {num_views} views: "
                f"Memory={result['memory_gb']['mean']:.2f}GB, "
                f"Freq={result['frequency_hz']['mean']:.2f}Hz"
            )

        except (RuntimeError, Exception) as e:
            error_str = str(e).lower()
            # Handle OOM and other CUDA resource limits (e.g., 32-bit index overflow)
            is_resource_limit = (
                "out of memory" in error_str
                or "32-bit index" in error_str
                or "cuda" in error_str
            )
            if is_resource_limit:
                reason = "OOM" if "out of memory" in error_str else "CUDA limit"
                print(
                    f"  {num_views} views: {reason} ({e}) - marking remaining as skipped"
                )
                torch.cuda.empty_cache()
                # Mark this and all remaining view counts as unable to run
                for remaining_views in num_views_list[i:]:
                    results[str(remaining_views)] = "oom"
                break
            else:
                # For other errors, just skip this view count but continue
                print(f"  {num_views} views: Error ({e}) - skipping")
                results[str(num_views)] = "error"

        # Clear cache between runs
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Plotting
# ============================================================================


def detect_overlapping_values(
    method_data_list: List[Tuple[str, List[int], List[float]]],
    tolerance: float = 0.01,
) -> Dict[int, List[List[int]]]:
    """
    Detect overlapping values between methods at each x-point.

    Args:
        method_data_list: List of tuples (method_name, x_values, y_values)
        tolerance: Relative tolerance for considering values as overlapping

    Returns:
        Dictionary mapping x_values to lists of overlapping method indices
    """
    overlaps = {}

    # Get all unique x values
    all_x_values = set()
    for _, x_values, _ in method_data_list:
        all_x_values.update(x_values)

    # For each x value, check for overlapping y values
    for x_val in all_x_values:
        y_values_at_x = []
        method_indices_at_x = []

        for i, (_, x_values, y_values) in enumerate(method_data_list):
            if x_val in x_values:
                x_idx = x_values.index(x_val)
                y_val = y_values[x_idx]
                y_values_at_x.append(y_val)
                method_indices_at_x.append(i)

        # Find overlapping groups using relative tolerance
        overlap_groups = []
        for i, y_val in enumerate(y_values_at_x):
            overlapping_indices = []
            for j, other_y_val in enumerate(y_values_at_x):
                # Use relative tolerance based on the magnitude of values
                rel_diff = abs(y_val - other_y_val) / max(abs(y_val), 1e-8)
                if rel_diff <= tolerance:
                    overlapping_indices.append(method_indices_at_x[j])

            if len(overlapping_indices) > 1:
                overlapping_indices.sort()
                if overlapping_indices not in overlap_groups:
                    overlap_groups.append(overlapping_indices)

        if overlap_groups:
            overlaps[x_val] = overlap_groups

    return overlaps


def get_overlap_style(
    method_idx: int,
    overlap_info: Dict[int, List[List[int]]],
    x_val: int,
) -> Tuple[float, float, float]:
    """
    Get visual style adjustments for overlapping values.

    Args:
        method_idx: Index of the current method
        overlap_info: Overlap information from detect_overlapping_values
        x_val: Current x value

    Returns:
        Tuple of (alpha, linewidth, markersize) adjustments
    """
    alpha = 0.9
    linewidth = 2.0
    markersize = 6.0

    if x_val in overlap_info:
        for overlap_group in overlap_info[x_val]:
            if method_idx in overlap_group:
                position_in_group = overlap_group.index(method_idx)
                group_size = len(overlap_group)

                if group_size > 1:
                    # Layered approach: earlier methods get thicker lines, more transparent
                    # Later methods get thinner lines, less transparent (on top)
                    if position_in_group == 0:
                        linewidth = 6.0
                        alpha = 0.4
                        markersize = 10.0
                    elif position_in_group == 1:
                        linewidth = 3.5
                        alpha = 0.7
                        markersize = 7.0
                    else:
                        linewidth = 2.0
                        alpha = 0.9
                        markersize = 5.0
                break

    return alpha, linewidth, markersize


def plot_memory_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    output_path: str,
    num_views_list: List[int],
):
    """
    Create memory usage plot.

    Args:
        results: Dictionary with profiling results per model
        output_path: Path to save the plot
        num_views_list: List of view counts for x-axis
    """
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Sort methods so MapAnything variants appear first in legend
    sorted_method_names = sort_methods_mapanything_first(list(results.keys()))

    # First pass: collect all method data for overlap detection
    method_data_list = []
    all_x_values_with_data = set()

    for method_name in sorted_method_names:
        method_results = results[method_name]
        x_values = []
        y_values = []

        for num_views in num_views_list:
            key = str(num_views)
            if key in method_results and method_results[key] not in ("oom", "error"):
                x_values.append(num_views)
                y_values.append(method_results[key]["memory_gb"]["mean"])
                all_x_values_with_data.add(num_views)

        if x_values:
            method_data_list.append((method_name, x_values, y_values))

    # Detect overlapping values
    overlap_info = detect_overlapping_values(method_data_list)

    # Second pass: plot with overlap-aware styling
    for method_idx, (method_name, x_values, y_values) in enumerate(method_data_list):
        color = get_method_color(method_name)
        display_name = get_method_display_name(method_name)

        # Determine if this method has any overlaps and get average style
        alphas = []
        linewidths = []
        markersizes = []
        for x_val in x_values:
            alpha, lw, ms = get_overlap_style(method_idx, overlap_info, x_val)
            alphas.append(alpha)
            linewidths.append(lw)
            markersizes.append(ms)

        # Use average style for the entire line
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0.9
        avg_linewidth = sum(linewidths) / len(linewidths) if linewidths else 2.0
        avg_markersize = sum(markersizes) / len(markersizes) if markersizes else 6.0

        ax.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            linestyle="-",
            label=display_name,
            linewidth=avg_linewidth,
            markersize=avg_markersize,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=avg_alpha,
        )

    ax.set_xlabel("Number of Views", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Peak GPU Memory (GB) $\downarrow$", fontsize=12, fontweight="bold")
    ax.set_title("GPU Memory Usage vs Number of Views", fontsize=14, fontweight="bold")

    # Log scale x-axis with only data point ticks (disable minor ticks)
    ax.set_xscale("log")
    x_ticks = sorted(all_x_values_with_data)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks])
    ax.set_xticks([], minor=True)  # Disable minor ticks

    # Grid and legend
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Memory plot saved to: {output_path}")
    plt.close()


def plot_speed_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    output_path: str,
    num_views_list: List[int],
):
    """
    Create inference speed (frequency) plot.

    Args:
        results: Dictionary with profiling results per model
        output_path: Path to save the plot
        num_views_list: List of view counts for x-axis
    """
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Sort methods so MapAnything variants appear first in legend
    sorted_method_names = sort_methods_mapanything_first(list(results.keys()))

    # First pass: collect all method data for overlap detection
    method_data_list = []
    all_x_values_with_data = set()

    for method_name in sorted_method_names:
        method_results = results[method_name]
        x_values = []
        y_values = []

        for num_views in num_views_list:
            key = str(num_views)
            if key in method_results and method_results[key] not in ("oom", "error"):
                x_values.append(num_views)
                y_values.append(method_results[key]["frequency_hz"]["mean"])
                all_x_values_with_data.add(num_views)

        if x_values:
            method_data_list.append((method_name, x_values, y_values))

    # Detect overlapping values
    overlap_info = detect_overlapping_values(method_data_list)

    # Second pass: plot with overlap-aware styling
    for method_idx, (method_name, x_values, y_values) in enumerate(method_data_list):
        color = get_method_color(method_name)
        display_name = get_method_display_name(method_name)

        # Determine if this method has any overlaps and get average style
        alphas = []
        linewidths = []
        markersizes = []
        for x_val in x_values:
            alpha, lw, ms = get_overlap_style(method_idx, overlap_info, x_val)
            alphas.append(alpha)
            linewidths.append(lw)
            markersizes.append(ms)

        # Use average style for the entire line
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0.9
        avg_linewidth = sum(linewidths) / len(linewidths) if linewidths else 2.0
        avg_markersize = sum(markersizes) / len(markersizes) if markersizes else 6.0

        ax.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            linestyle="-",
            label=display_name,
            linewidth=avg_linewidth,
            markersize=avg_markersize,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=avg_alpha,
        )

    ax.set_xlabel("Number of Views", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        r"Inference Frequency (Hz) - Log Scale $\uparrow$",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Inference Speed vs Number of Views",
        fontsize=14,
        fontweight="bold",
    )

    # Log scale x-axis with only data point ticks (disable minor ticks)
    ax.set_xscale("log")
    x_ticks = sorted(all_x_values_with_data)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks])
    ax.set_xticks([], minor=True)  # Disable minor ticks

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Grid and legend
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Speed plot saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile GPU memory and inference speed of MapAnything and external models."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save profiling results (JSON and plots)",
    )

    parser.add_argument(
        "--num_views",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 24, 32, 50, 100, 200, 500],
        help="List of view counts to profile (default: 2 4 8 16 24 32 50 100 200 500)",
    )

    parser.add_argument(
        "--external_models",
        type=str,
        nargs="*",
        default=[],
        help="External model names to compare (e.g., vggt pi3x must3r). "
        "Models are loaded from configs/model/<name>.yaml",
    )

    parser.add_argument(
        "--mapanything_checkpoint",
        type=str,
        default=None,
        help="Path to MapAnything checkpoint. If not provided, loads from HuggingFace.",
    )

    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed MapAnything model (only when loading from HuggingFace)",
    )

    parser.add_argument(
        "--profile_v1",
        action="store_true",
        help="Also profile MapAnything V1 models (deprecated but available for comparison)",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warmup iterations before timing (default: 3)",
    )

    parser.add_argument(
        "--timed_runs",
        type=int,
        default=5,
        help="Number of timed iterations for statistics (default: 5)",
    )

    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)",
    )

    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="AMP dtype (default: bf16)",
    )

    parser.add_argument(
        "--skip_mem_efficient",
        action="store_true",
        help="Skip memory-efficient MapAnything profiling",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing profiling_results.json, skipping already-profiled view counts",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling. No GPU detected.")

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")

    # Check for existing results to resume from
    json_path = os.path.join(args.output_dir, "profiling_results.json")
    all_results = {}

    if args.resume and os.path.exists(json_path):
        print("\n" + "=" * 60)
        print("Resuming from existing results")
        print("=" * 60)
        with open(json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        all_results = existing_data.get("results", {})
        print(f"Loaded existing results for: {list(all_results.keys())}")

    # Helper to check if all view counts are already profiled
    def all_views_profiled(model_name: str) -> bool:
        existing = all_results.get(model_name, {})
        return all(str(v) in existing for v in args.num_views)

    # ========================================================================
    # Profile MapAnything (default mode and memory-efficient mode)
    # ========================================================================
    # Check if we need to load MapAnything at all
    need_mapanything = not all_views_profiled("mapanything")
    need_mem_efficient = not args.skip_mem_efficient and not all_views_profiled(
        "mapanything_mem_efficient"
    )

    if need_mapanything or need_mem_efficient:
        print("\n" + "=" * 60)
        print("Loading MapAnything model")
        print("=" * 60)

        model, norm_type, resolution = load_mapanything_from_pretrained(
            device=device,
            checkpoint_path=args.mapanything_checkpoint,
            use_apache=args.apache,
        )

        # Profile default mode
        if need_mapanything:
            print("\n" + "=" * 60)
            print("Profiling MapAnything (default mode)")
            print("=" * 60)

            all_results["mapanything"] = profile_model_across_views(
                model=model,
                model_name="mapanything",
                data_norm_type=norm_type,
                resolution=resolution,
                num_views_list=args.num_views,
                device=device,
                num_warmup=args.warmup_runs,
                num_timed=args.timed_runs,
                memory_efficient=False,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                existing_results=all_results.get("mapanything"),
            )
        else:
            print("\n[MapAnything default] All view counts already profiled, skipping")

        # Profile memory-efficient mode
        if need_mem_efficient:
            print("\n" + "=" * 60)
            print("Profiling MapAnything (memory-efficient mode)")
            print("=" * 60)

            torch.cuda.empty_cache()

            all_results["mapanything_mem_efficient"] = profile_model_across_views(
                model=model,
                model_name="mapanything_mem_efficient",
                data_norm_type=norm_type,
                resolution=resolution,
                num_views_list=args.num_views,
                device=device,
                num_warmup=args.warmup_runs,
                num_timed=args.timed_runs,
                memory_efficient=True,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                existing_results=all_results.get("mapanything_mem_efficient"),
            )
        elif not args.skip_mem_efficient:
            print(
                "\n[MapAnything mem-efficient] All view counts already profiled, skipping"
            )

        # Free MapAnything model memory
        del model
        torch.cuda.empty_cache()
    else:
        print(
            "\n[MapAnything] All view counts already profiled, skipping model loading"
        )

    # ========================================================================
    # Profile MapAnything V1 (if requested)
    # ========================================================================
    if args.profile_v1:
        need_v1 = not all_views_profiled("mapanything_v1")
        need_v1_mem_efficient = not args.skip_mem_efficient and not all_views_profiled(
            "mapanything_v1_mem_efficient"
        )

        if need_v1 or need_v1_mem_efficient:
            print("\n" + "=" * 60)
            print("Loading MapAnything V1 model")
            print("=" * 60)

            model_v1, norm_type_v1, resolution_v1 = load_mapanything_from_pretrained(
                device=device,
                checkpoint_path=None,  # V1 always loads from HuggingFace
                use_apache=args.apache,
                use_v1=True,
            )

            # Profile V1 default mode
            if need_v1:
                print("\n" + "=" * 60)
                print("Profiling MapAnything V1 (default mode)")
                print("=" * 60)

                all_results["mapanything_v1"] = profile_model_across_views(
                    model=model_v1,
                    model_name="mapanything_v1",
                    data_norm_type=norm_type_v1,
                    resolution=resolution_v1,
                    num_views_list=args.num_views,
                    device=device,
                    num_warmup=args.warmup_runs,
                    num_timed=args.timed_runs,
                    memory_efficient=False,
                    use_amp=args.use_amp,
                    amp_dtype=args.amp_dtype,
                    existing_results=all_results.get("mapanything_v1"),
                )
            else:
                print(
                    "\n[MapAnything V1 default] All view counts already profiled, skipping"
                )

            # Profile V1 memory-efficient mode
            if need_v1_mem_efficient:
                print("\n" + "=" * 60)
                print("Profiling MapAnything V1 (memory-efficient mode)")
                print("=" * 60)

                torch.cuda.empty_cache()

                all_results["mapanything_v1_mem_efficient"] = (
                    profile_model_across_views(
                        model=model_v1,
                        model_name="mapanything_v1_mem_efficient",
                        data_norm_type=norm_type_v1,
                        resolution=resolution_v1,
                        num_views_list=args.num_views,
                        device=device,
                        num_warmup=args.warmup_runs,
                        num_timed=args.timed_runs,
                        memory_efficient=True,
                        use_amp=args.use_amp,
                        amp_dtype=args.amp_dtype,
                        existing_results=all_results.get(
                            "mapanything_v1_mem_efficient"
                        ),
                    )
                )
            elif not args.skip_mem_efficient:
                print(
                    "\n[MapAnything V1 mem-efficient] All view counts already profiled, skipping"
                )

            # Free V1 model memory
            del model_v1
            torch.cuda.empty_cache()
        else:
            print(
                "\n[MapAnything V1] All view counts already profiled, skipping model loading"
            )

    # ========================================================================
    # Profile External Models
    # ========================================================================
    for external_model_name in args.external_models:
        # Check if all view counts are already profiled for this model
        if all_views_profiled(external_model_name):
            print(
                f"\n[{external_model_name}] All view counts already profiled, skipping"
            )
            continue

        print("\n" + "=" * 60)
        print(f"Profiling {external_model_name}")
        print("=" * 60)

        try:
            ext_model, ext_norm_type, ext_resolution = load_model_with_metadata(
                model_name=external_model_name,
                device=device,
            )

            all_results[external_model_name] = profile_model_across_views(
                model=ext_model,
                model_name=external_model_name,
                data_norm_type=ext_norm_type,
                resolution=ext_resolution,
                num_views_list=args.num_views,
                device=device,
                num_warmup=args.warmup_runs,
                num_timed=args.timed_runs,
                memory_efficient=False,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                existing_results=all_results.get(external_model_name),
            )

            # Free model memory
            del ext_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error profiling {external_model_name}: {e}")
            continue

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    # Prepare output data
    output_data = {
        "metadata": {
            "device": gpu_name,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "use_amp": args.use_amp,
            "amp_dtype": args.amp_dtype,
            "timestamp": datetime.now().isoformat(),
            "num_views_profiled": args.num_views,
        },
        "results": all_results,
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, "profiling_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate plots
    memory_plot_path = os.path.join(args.output_dir, "profiling_memory.png")
    speed_plot_path = os.path.join(args.output_dir, "profiling_speed.png")

    plot_memory_results(all_results, memory_plot_path, args.num_views)
    plot_speed_results(all_results, speed_plot_path, args.num_views)

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
