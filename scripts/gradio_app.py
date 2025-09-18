# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script for Gradio App. Reflects the demo hosted on HuggingFace Spaces.
"""

import gc
import os
import shutil
import sys
import time
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from pillow_heif import register_heif_opener

from mapanything.utils.geometry import depthmap_to_world_frame, points_to_normals
from mapanything.utils.hf_utils.css_and_html import (
    get_acknowledgements_html,
    get_description_html,
    get_gradio_theme,
    get_header_html,
    GRADIO_CSS,
    MEASURE_INSTRUCTIONS_HTML,
)
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.image import load_images, rgb

register_heif_opener()

sys.path.append("mapanything/")


def get_logo_base64():
    """Convert WAI logo to base64 for embedding in HTML"""
    import base64

    logo_path = "examples/WAI-Logo/wai_logo.png"
    try:
        with open(logo_path, "rb") as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{base64_str}"
    except FileNotFoundError:
        return None


# MapAnything Configuration
high_level_config = {
    "path": "configs/train.yaml",
    "hf_model_name": "facebook/map-anything",
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_name": "model.safetensors",
    "config_name": "config.json",
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
}


# Initialize model - this will be done on GPU when needed
model = None


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_model(
    target_dir,
    apply_mask=True,
    mask_edges=True,
    filter_black_bg=False,
    filter_white_bg=False,
):
    """
    Run the MapAnything model on images in the 'target_dir/images' folder and return predictions.
    """
    global model
    import torch  # Ensure torch is available in function scope

    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize model if not already done
    if model is None:
        model = initialize_mapanything_model(high_level_config, device)

    else:
        model = model.to(device)

    model.eval()

    # Load images using MapAnything's load_images function
    print("Loading images...")
    image_folder_path = os.path.join(target_dir, "images")
    views = load_images(image_folder_path)

    print(f"Loaded {len(views)} images")
    if len(views) == 0:
        raise ValueError("No images found. Check your upload.")

    # Run model inference
    print("Running inference...")
    # apply_mask: Whether to apply the non-ambiguous mask to the output. Defaults to True.
    # mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
    # Use checkbox values - mask_edges is set to True by default since there's no UI control for it
    outputs = model.infer(
        views, apply_mask=apply_mask, mask_edges=True, memory_efficient_inference=False
    )

    # Convert predictions to format expected by visualization
    predictions = {}

    # Initialize lists for the required keys
    extrinsic_list = []
    intrinsic_list = []
    world_points_list = []
    depth_maps_list = []
    images_list = []
    final_mask_list = []
    confidences = []
    # Loop through the outputs
    for pred in outputs:
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        conf = pred["conf"][0].squeeze(-1)  # (H, W)
        # Compute new pts3d using depth, intrinsics, and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays for visualization
        # Check if mask key exists in pred, if not, fill with boolean trues in the size of depthmap_torch
        if "mask" in pred:
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            # Fill with boolean trues in the size of depthmap_torch
            mask = np.ones_like(depthmap_torch.cpu().numpy(), dtype=bool)

        # Combine with valid depth mask
        mask = mask & valid_mask.cpu().numpy()

        image = pred["img_no_norm"][0].cpu().numpy()

        # Append to lists
        extrinsic_list.append(camera_pose_torch.cpu().numpy())
        intrinsic_list.append(intrinsics_torch.cpu().numpy())
        world_points_list.append(pts3d_computed.cpu().numpy())
        depth_maps_list.append(depthmap_torch.cpu().numpy())
        images_list.append(image)  # Add image to list
        final_mask_list.append(mask)  # Add final_mask to list
        confidences.append(conf.cpu().numpy())  # Add confidence to list

    # Convert lists to numpy arrays with required shapes
    # extrinsic: (S, 3, 4) - batch of camera extrinsic matrices
    predictions["extrinsic"] = np.stack(extrinsic_list, axis=0)

    # intrinsic: (S, 3, 3) - batch of camera intrinsic matrices
    predictions["intrinsic"] = np.stack(intrinsic_list, axis=0)

    # world_points: (S, H, W, 3) - batch of 3D world points
    predictions["world_points"] = np.stack(world_points_list, axis=0)

    predictions["conf"] = np.stack(confidences, axis=0)

    # depth: (S, H, W, 1) or (S, H, W) - batch of depth maps
    depth_maps = np.stack(depth_maps_list, axis=0)
    # Add channel dimension if needed to match (S, H, W, 1) format
    if len(depth_maps.shape) == 3:
        depth_maps = depth_maps[..., np.newaxis]

    predictions["depth"] = depth_maps

    # images: (S, H, W, 3) - batch of input images
    predictions["images"] = np.stack(images_list, axis=0)

    # final_mask: (S, H, W) - batch of final masks for filtering
    predictions["final_mask"] = np.stack(final_mask_list, axis=0)

    # Process data for visualization tabs (depth, normal, measure)
    processed_data = process_predictions_for_visualization(
        predictions, views, high_level_config, filter_black_bg, filter_white_bg
    )

    # Clean up
    torch.cuda.empty_cache()

    return predictions, processed_data


def update_view_selectors(processed_data):
    """Update view selector dropdowns based on available views"""
    if processed_data is None or len(processed_data) == 0:
        choices = ["View 1"]
    else:
        num_views = len(processed_data)
        choices = [f"View {i + 1}" for i in range(num_views)]

    return (
        gr.Dropdown(choices=choices, value=choices[0]),  # depth_view_selector
        gr.Dropdown(choices=choices, value=choices[0]),  # normal_view_selector
        gr.Dropdown(choices=choices, value=choices[0]),  # measure_view_selector
    )


def get_view_data_by_index(processed_data, view_index):
    """Get view data by index, handling bounds"""
    if processed_data is None or len(processed_data) == 0:
        return None

    view_keys = list(processed_data.keys())
    if view_index < 0 or view_index >= len(view_keys):
        view_index = 0

    return processed_data[view_keys[view_index]]


def update_depth_view(processed_data, view_index):
    """Update depth view for a specific view index"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["depth"] is None:
        return None

    return colorize_depth(view_data["depth"], mask=view_data.get("mask"))


def update_normal_view(processed_data, view_index):
    """Update normal view for a specific view index"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["normal"] is None:
        return None

    return colorize_normal(view_data["normal"], mask=view_data.get("mask"))


def update_measure_view(processed_data, view_index):
    """Update measure view for a specific view index with mask overlay"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None:
        return None, []  # image, measure_points

    # Get the base image
    image = view_data["image"].copy()

    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Apply mask overlay if mask is available
    if view_data["mask"] is not None:
        mask = view_data["mask"]

        # Create light grey overlay for masked areas
        # Masked areas (False values) will be overlaid with light grey
        invalid_mask = ~mask  # Areas where mask is False

        if invalid_mask.any():
            # Create a light grey overlay (RGB: 192, 192, 192)
            overlay_color = np.array([255, 220, 220], dtype=np.uint8)

            # Apply overlay with some transparency
            alpha = 0.5  # Transparency level
            for c in range(3):  # RGB channels
                image[:, :, c] = np.where(
                    invalid_mask,
                    (1 - alpha) * image[:, :, c] + alpha * overlay_color[c],
                    image[:, :, c],
                ).astype(np.uint8)

    return image, []


def navigate_depth_view(processed_data, current_selector_value, direction):
    """Navigate depth view (direction: -1 for previous, +1 for next)"""
    if processed_data is None or len(processed_data) == 0:
        return "View 1", None

    # Parse current view number
    try:
        current_view = int(current_selector_value.split()[1]) - 1
    except:  # noqa
        current_view = 0

    num_views = len(processed_data)
    new_view = (current_view + direction) % num_views

    new_selector_value = f"View {new_view + 1}"
    depth_vis = update_depth_view(processed_data, new_view)

    return new_selector_value, depth_vis


def navigate_normal_view(processed_data, current_selector_value, direction):
    """Navigate normal view (direction: -1 for previous, +1 for next)"""
    if processed_data is None or len(processed_data) == 0:
        return "View 1", None

    # Parse current view number
    try:
        current_view = int(current_selector_value.split()[1]) - 1
    except:  # noqa
        current_view = 0

    num_views = len(processed_data)
    new_view = (current_view + direction) % num_views

    new_selector_value = f"View {new_view + 1}"
    normal_vis = update_normal_view(processed_data, new_view)

    return new_selector_value, normal_vis


def navigate_measure_view(processed_data, current_selector_value, direction):
    """Navigate measure view (direction: -1 for previous, +1 for next)"""
    if processed_data is None or len(processed_data) == 0:
        return "View 1", None, []

    # Parse current view number
    try:
        current_view = int(current_selector_value.split()[1]) - 1
    except:  # noqa
        current_view = 0

    num_views = len(processed_data)
    new_view = (current_view + direction) % num_views

    new_selector_value = f"View {new_view + 1}"
    measure_image, measure_points = update_measure_view(processed_data, new_view)

    return new_selector_value, measure_image, measure_points


def populate_visualization_tabs(processed_data):
    """Populate the depth, normal, and measure tabs with processed data"""
    if processed_data is None or len(processed_data) == 0:
        return None, None, None, []

    # Use update functions to ensure confidence filtering is applied from the start
    depth_vis = update_depth_view(processed_data, 0)
    normal_vis = update_normal_view(processed_data, 0)
    measure_img, _ = update_measure_view(processed_data, 0)

    return depth_vis, normal_vis, measure_img, []


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, s_time_interval=1.0):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data

            # Check if the file is a HEIC image
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".heic", ".heif"]:
                # Convert HEIC to JPEG for better gallery compatibility
                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB if necessary (HEIC can have different color modes)
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")

                        # Create JPEG filename
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        dst_path = os.path.join(target_dir_images, f"{base_name}.jpg")

                        # Save as JPEG with high quality
                        img.save(dst_path, "JPEG", quality=95)
                        image_paths.append(dst_path)
                        print(
                            f"Converted HEIC to JPEG: {os.path.basename(file_path)} -> {os.path.basename(dst_path)}"
                        )
                except Exception as e:
                    print(f"Error converting HEIC file {file_path}: {e}")
                    # Fall back to copying as is
                    dst_path = os.path.join(
                        target_dir_images, os.path.basename(file_path)
                    )
                    shutil.copy(file_path, dst_path)
                    image_paths.append(dst_path)
            else:
                # Regular image files - copy as is
                dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
                shutil.copy(file_path, dst_path)
                image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * s_time_interval)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(
                    target_dir_images, f"{video_frame_num:06}.png"
                )
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(
        f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds"
    )
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, s_time_interval=1.0):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, s_time_interval)
    return (
        None,
        target_dir,
        image_paths,
        "Upload complete. Click 'Reconstruct' to begin 3D processing.",
    )


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
@spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    frame_filter="All",
    show_cam=True,
    filter_black_bg=False,
    filter_white_bg=False,
    conf_thres=3.0,
    apply_mask=True,
    show_mesh=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = (
        sorted(os.listdir(target_dir_images))
        if os.path.isdir(target_dir_images)
        else []
    )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running MapAnything model...")
    with torch.no_grad():
        predictions, processed_data = run_model(target_dir, apply_mask)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_mesh{show_mesh}_black{filter_black_bg}_white{filter_white_bg}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
        mask_black_bg=filter_black_bg,
        mask_white_bg=filter_white_bg,
        as_mesh=show_mesh,  # Use the show_mesh parameter
        conf_percentile=conf_thres,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    log_msg = (
        f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
    )

    # Populate visualization tabs with processed data
    depth_vis, normal_vis, measure_img, measure_pts = populate_visualization_tabs(
        processed_data
    )

    # Update view selectors based on available views
    depth_selector, normal_selector, measure_selector = update_view_selectors(
        processed_data
    )

    return (
        glbfile,
        log_msg,
        gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True),
        processed_data,
        depth_vis,
        normal_vis,
        measure_img,
        "",  # measure_text (empty initially)
        depth_selector,
        normal_selector,
        measure_selector,
    )


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def colorize_depth(depth_map, mask=None):
    """Convert depth map to colorized visualization with optional mask"""
    if depth_map is None:
        return None

    # Normalize depth to 0-1 range
    depth_normalized = depth_map.copy()
    valid_mask = depth_normalized > 0

    # Apply additional mask if provided (for background filtering)
    if mask is not None:
        valid_mask = valid_mask & mask

    if valid_mask.sum() > 0:
        valid_depths = depth_normalized[valid_mask]
        p5 = np.percentile(valid_depths, 5)
        p95 = np.percentile(valid_depths, 95)

        depth_normalized[valid_mask] = (depth_normalized[valid_mask] - p5) / (p95 - p5)

    # Apply colormap
    import matplotlib.pyplot as plt

    colormap = plt.cm.turbo_r
    colored = colormap(depth_normalized)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    # Set invalid pixels to white
    colored[~valid_mask] = [255, 255, 255]

    return colored


def colorize_normal(normal_map, mask=None):
    """Convert normal map to colorized visualization with optional mask"""
    if normal_map is None:
        return None

    # Create a copy for modification
    normal_vis = normal_map.copy()

    # Apply mask if provided (set masked areas to [0, 0, 0] which becomes grey after normalization)
    if mask is not None:
        invalid_mask = ~mask
        normal_vis[invalid_mask] = [0, 0, 0]  # Set invalid areas to zero

    # Normalize normals to [0, 1] range for visualization
    normal_vis = (normal_vis + 1.0) / 2.0
    normal_vis = (normal_vis * 255).astype(np.uint8)

    return normal_vis


def process_predictions_for_visualization(
    predictions, views, high_level_config, filter_black_bg=False, filter_white_bg=False
):
    """Extract depth, normal, and 3D points from predictions for visualization"""
    processed_data = {}

    # Process each view
    for view_idx, view in enumerate(views):
        # Get image
        image = rgb(view["img"], norm_type=high_level_config["data_norm_type"])

        # Get predicted points
        pred_pts3d = predictions["world_points"][view_idx]

        # Initialize data for this view
        view_data = {
            "image": image[0],
            "points3d": pred_pts3d,
            "depth": None,
            "normal": None,
            "mask": None,
        }

        # Start with the final mask from predictions
        mask = predictions["final_mask"][view_idx].copy()

        # Apply black background filtering if enabled
        if filter_black_bg:
            # Get the image colors (ensure they're in 0-255 range)
            view_colors = image[0] * 255 if image[0].max() <= 1.0 else image[0]
            # Filter out black background pixels (sum of RGB < 16)
            black_bg_mask = view_colors.sum(axis=2) >= 16
            mask = mask & black_bg_mask

        # Apply white background filtering if enabled
        if filter_white_bg:
            # Get the image colors (ensure they're in 0-255 range)
            view_colors = image[0] * 255 if image[0].max() <= 1.0 else image[0]
            # Filter out white background pixels (all RGB > 240)
            white_bg_mask = ~(
                (view_colors[:, :, 0] > 240)
                & (view_colors[:, :, 1] > 240)
                & (view_colors[:, :, 2] > 240)
            )
            mask = mask & white_bg_mask

        view_data["mask"] = mask
        view_data["depth"] = predictions["depth"][view_idx].squeeze()

        normals, _ = points_to_normals(pred_pts3d, mask=view_data["mask"])
        view_data["normal"] = normals

        processed_data[view_idx] = view_data

    return processed_data


def reset_measure(processed_data):
    """Reset measure points"""
    if processed_data is None or len(processed_data) == 0:
        return None, [], ""

    # Return the first view image
    first_view = list(processed_data.values())[0]
    return first_view["image"], [], ""


def measure(
    processed_data, measure_points, current_view_selector, event: gr.SelectData
):
    """Handle measurement on images"""
    try:
        print(f"Measure function called with selector: {current_view_selector}")

        if processed_data is None or len(processed_data) == 0:
            return None, [], "No data available"

        # Use the currently selected view instead of always using the first view
        try:
            current_view_index = int(current_view_selector.split()[1]) - 1
        except:  # noqa
            current_view_index = 0

        print(f"Using view index: {current_view_index}")

        # Get view data safely
        if current_view_index < 0 or current_view_index >= len(processed_data):
            current_view_index = 0

        view_keys = list(processed_data.keys())
        current_view = processed_data[view_keys[current_view_index]]

        if current_view is None:
            return None, [], "No view data available"

        point2d = event.index[0], event.index[1]
        print(f"Clicked point: {point2d}")

        # Check if the clicked point is in a masked area (prevent interaction)
        if (
            current_view["mask"] is not None
            and 0 <= point2d[1] < current_view["mask"].shape[0]
            and 0 <= point2d[0] < current_view["mask"].shape[1]
        ):
            # Check if the point is in a masked (invalid) area
            if not current_view["mask"][point2d[1], point2d[0]]:
                print(f"Clicked point {point2d} is in masked area, ignoring click")
                # Always return image with mask overlay
                masked_image, _ = update_measure_view(
                    processed_data, current_view_index
                )
                return (
                    masked_image,
                    measure_points,
                    '<span style="color: red; font-weight: bold;">Cannot measure on masked areas (shown in grey)</span>',
                )

        measure_points.append(point2d)

        # Get image with mask overlay and ensure it's valid
        image, _ = update_measure_view(processed_data, current_view_index)
        if image is None:
            return None, [], "No image available"

        image = image.copy()
        points3d = current_view["points3d"]

        # Ensure image is in uint8 format for proper cv2 operations
        try:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    # Image is in [0, 1] range, convert to [0, 255]
                    image = (image * 255).astype(np.uint8)
                else:
                    # Image is already in [0, 255] range
                    image = image.astype(np.uint8)
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None, [], f"Image conversion error: {e}"

        # Draw circles for points
        try:
            for p in measure_points:
                if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                    image = cv2.circle(
                        image, p, radius=5, color=(255, 0, 0), thickness=2
                    )
        except Exception as e:
            print(f"Drawing error: {e}")
            return None, [], f"Drawing error: {e}"

        depth_text = ""
        try:
            for i, p in enumerate(measure_points):
                if (
                    current_view["depth"] is not None
                    and 0 <= p[1] < current_view["depth"].shape[0]
                    and 0 <= p[0] < current_view["depth"].shape[1]
                ):
                    d = current_view["depth"][p[1], p[0]]
                    depth_text += f"- **P{i + 1} depth: {d:.2f}m.**\n"
                else:
                    # Use Z coordinate of 3D points if depth not available
                    if (
                        points3d is not None
                        and 0 <= p[1] < points3d.shape[0]
                        and 0 <= p[0] < points3d.shape[1]
                    ):
                        z = points3d[p[1], p[0], 2]
                        depth_text += f"- **P{i + 1} Z-coord: {z:.2f}m.**\n"
        except Exception as e:
            print(f"Depth text error: {e}")
            depth_text = f"Error computing depth: {e}\n"

        if len(measure_points) == 2:
            try:
                point1, point2 = measure_points
                # Draw line
                if (
                    0 <= point1[0] < image.shape[1]
                    and 0 <= point1[1] < image.shape[0]
                    and 0 <= point2[0] < image.shape[1]
                    and 0 <= point2[1] < image.shape[0]
                ):
                    image = cv2.line(
                        image, point1, point2, color=(255, 0, 0), thickness=2
                    )

                # Compute 3D distance
                distance_text = "- **Distance: Unable to compute**"
                if (
                    points3d is not None
                    and 0 <= point1[1] < points3d.shape[0]
                    and 0 <= point1[0] < points3d.shape[1]
                    and 0 <= point2[1] < points3d.shape[0]
                    and 0 <= point2[0] < points3d.shape[1]
                ):
                    try:
                        p1_3d = points3d[point1[1], point1[0]]
                        p2_3d = points3d[point2[1], point2[0]]
                        distance = np.linalg.norm(p1_3d - p2_3d)
                        distance_text = f"- **Distance: {distance:.2f}m**"
                    except Exception as e:
                        print(f"Distance computation error: {e}")
                        distance_text = f"- **Distance computation error: {e}**"

                measure_points = []
                text = depth_text + distance_text
                print(f"Measurement complete: {text}")
                return [image, measure_points, text]
            except Exception as e:
                print(f"Final measurement error: {e}")
                return None, [], f"Measurement error: {e}"
        else:
            print(f"Single point measurement: {depth_text}")
            return [image, measure_points, depth_text]

    except Exception as e:
        print(f"Overall measure function error: {e}")
        return None, [], f"Measure function error: {e}"


def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir,
    frame_filter,
    show_cam,
    is_example,
    conf_thres=None,
    filter_black_bg=False,
    filter_white_bg=False,
    show_mesh=True,
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return (
            gr.update(),
            "No reconstruction available. Please click the Reconstruct button first.",
        )

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return (
            gr.update(),
            "No reconstruction available. Please click the Reconstruct button first.",
        )

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return (
            gr.update(),
            f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first.",
        )

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_mesh{show_mesh}_black{filter_black_bg}_white{filter_white_bg}.glb",
    )

    
    glbscene = predictions_to_glb(
            predictions,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
            mask_black_bg=filter_black_bg,
            mask_white_bg=filter_white_bg,
            as_mesh=show_mesh,
            conf_percentile=conf_thres,
        )
    glbscene.export(file_obj=glbfile)

    return (
        glbfile,
        "Visualization updated.",
    )


def update_all_views_on_filter_change(
    target_dir,
    filter_black_bg,
    filter_white_bg,
    processed_data,
    depth_view_selector,
    normal_view_selector,
    measure_view_selector,
):
    """
    Update all individual view tabs when background filtering checkboxes change.
    This regenerates the processed data with new filtering and updates all views.
    """
    # Check if we have a valid target directory and predictions
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return processed_data, None, None, None, []

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return processed_data, None, None, None, []

    try:
        # Load the original predictions and views
        loaded = np.load(predictions_path, allow_pickle=True)
        predictions = {key: loaded[key] for key in loaded.keys()}

        # Load images using MapAnything's load_images function
        image_folder_path = os.path.join(target_dir, "images")
        views = load_images(image_folder_path)

        # Regenerate processed data with new filtering settings
        new_processed_data = process_predictions_for_visualization(
            predictions, views, high_level_config, filter_black_bg, filter_white_bg
        )

        # Get current view indices
        try:
            depth_view_idx = (
                int(depth_view_selector.split()[1]) - 1 if depth_view_selector else 0
            )
        except:  # noqa
            depth_view_idx = 0

        try:
            normal_view_idx = (
                int(normal_view_selector.split()[1]) - 1 if normal_view_selector else 0
            )
        except:  # noqa
            normal_view_idx = 0

        try:
            measure_view_idx = (
                int(measure_view_selector.split()[1]) - 1
                if measure_view_selector
                else 0
            )
        except:  # noqa
            measure_view_idx = 0

        # Update all views with new filtered data
        depth_vis = update_depth_view(new_processed_data, depth_view_idx)
        normal_vis = update_normal_view(new_processed_data, normal_view_idx)
        measure_img, _ = update_measure_view(new_processed_data, measure_view_idx)

        return new_processed_data, depth_vis, normal_vis, measure_img, []

    except Exception as e:
        print(f"Error updating views on filter change: {e}")
        return processed_data, None, None, None, []


# -------------------------------------------------------------------------
# Example scene functions
# -------------------------------------------------------------------------
def get_scene_info(examples_dir):
    """Get information about scenes in the examples directory"""
    import glob

    scenes = []
    if not os.path.exists(examples_dir):
        return scenes

    for scene_folder in sorted(os.listdir(examples_dir)):
        scene_path = os.path.join(examples_dir, scene_folder)
        if os.path.isdir(scene_path):
            # Find all image files in the scene folder
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(scene_path, ext)))
                image_files.extend(glob.glob(os.path.join(scene_path, ext.upper())))

            if image_files:
                # Sort images and get the first one for thumbnail
                image_files = sorted(image_files)
                first_image = image_files[0]
                num_images = len(image_files)

                scenes.append(
                    {
                        "name": scene_folder,
                        "path": scene_path,
                        "thumbnail": first_image,
                        "num_images": num_images,
                        "image_files": image_files,
                    }
                )

    return scenes


def load_example_scene(scene_name, examples_dir="examples"):
    """Load a scene from examples directory"""
    scenes = get_scene_info(examples_dir)

    # Find the selected scene
    selected_scene = None
    for scene in scenes:
        if scene["name"] == scene_name:
            selected_scene = scene
            break

    if selected_scene is None:
        return None, None, None, "Scene not found"

    # Create target directory and copy images
    target_dir, image_paths = handle_uploads(None, selected_scene["image_files"])

    return (
        None,  # Clear reconstruction output
        target_dir,  # Set target directory
        image_paths,  # Set gallery
        f"Loaded scene '{scene_name}' with {selected_scene['num_images']} images. Click 'Reconstruct' to begin 3D processing.",
    )


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = get_gradio_theme()

with gr.Blocks(theme=theme, css=GRADIO_CSS) as demo:
    # State variables for the tabbed interface
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    processed_data_state = gr.State(value=None)
    measure_points_state = gr.State(value=[])
    current_view_index = gr.State(value=0)  # Track current view index for navigation

    gr.HTML(get_header_html(get_logo_base64()))
    gr.HTML(get_description_html())

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            s_time_interval = gr.Slider(
                minimum=0.1,
                maximum=5.0,
                value=1.0,
                step=0.1,
                label="Sample time interval (take a sample every x sec.)",
                interactive=True,
                visible=True,
            )
            input_images = gr.File(
                file_count="multiple", label="Upload Images", interactive=True
            )

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown(
                    "**Metric 3D Reconstruction (Point Cloud and Camera Poses)**"
                )
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.",
                    elem_classes=["custom-log"],
                )

                # Add tabbed interface similar to MoGe
                with gr.Tabs():
                    with gr.Tab("3D View"):
                        reconstruction_output = gr.Model3D(
                            height=520,
                            zoom_speed=0.5,
                            pan_speed=0.5,
                            clear_color=[0.0, 0.0, 0.0, 0.0],
                            key="persistent_3d_viewer",
                            elem_id="reconstruction_3d_viewer",
                        )
                    with gr.Tab("Depth"):
                        with gr.Row(elem_classes=["navigation-row"]):
                            prev_depth_btn = gr.Button("◀ Previous", size="sm", scale=1)
                            depth_view_selector = gr.Dropdown(
                                choices=["View 1"],
                                value="View 1",
                                label="Select View",
                                scale=2,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            next_depth_btn = gr.Button("Next ▶", size="sm", scale=1)
                        depth_map = gr.Image(
                            type="numpy",
                            label="Colorized Depth Map",
                            format="png",
                            interactive=False,
                        )
                    with gr.Tab("Normal"):
                        with gr.Row(elem_classes=["navigation-row"]):
                            prev_normal_btn = gr.Button(
                                "◀ Previous", size="sm", scale=1
                            )
                            normal_view_selector = gr.Dropdown(
                                choices=["View 1"],
                                value="View 1",
                                label="Select View",
                                scale=2,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            next_normal_btn = gr.Button("Next ▶", size="sm", scale=1)
                        normal_map = gr.Image(
                            type="numpy",
                            label="Normal Map",
                            format="png",
                            interactive=False,
                        )
                    with gr.Tab("Measure"):
                        gr.Markdown(MEASURE_INSTRUCTIONS_HTML)
                        with gr.Row(elem_classes=["navigation-row"]):
                            prev_measure_btn = gr.Button(
                                "◀ Previous", size="sm", scale=1
                            )
                            measure_view_selector = gr.Dropdown(
                                choices=["View 1"],
                                value="View 1",
                                label="Select View",
                                scale=2,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            next_measure_btn = gr.Button("Next ▶", size="sm", scale=1)
                        measure_image = gr.Image(
                            type="numpy",
                            show_label=False,
                            format="webp",
                            interactive=False,
                            sources=[],
                        )
                        gr.Markdown(
                            "**Note:** Light-grey areas indicate regions with no depth information where measurements cannot be taken."
                        )
                        measure_text = gr.Markdown("")

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [
                        input_video,
                        input_images,
                        reconstruction_output,
                        log_output,
                        target_dir_output,
                        image_gallery,
                    ],
                    scale=1,
                )

            with gr.Row():
                frame_filter = gr.Dropdown(
                    choices=["All"], value="All", label="Show Points from Frame"
                )
                with gr.Column():
                    gr.Markdown("### Pointcloud Options: (live updates)")
                    conf_thres = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=0.1,
                        label="Confidence Threshold Percentile (mask 3D points below this)",
                        interactive=True,
                    )

                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    show_mesh = gr.Checkbox(label="Show Mesh", value=True)
                    filter_black_bg = gr.Checkbox(
                        label="Filter Black Background", value=False
                    )
                    filter_white_bg = gr.Checkbox(
                        label="Filter White Background", value=False
                    )
                    gr.Markdown("### Reconstruction Options: (updated on next run)")
                    apply_mask_checkbox = gr.Checkbox(
                        label="Apply mask for predicted ambiguous depth classes & edges",
                        value=True,
                    )
    # ---------------------- Example Scenes Section ----------------------
    gr.Markdown("## Example Scenes (lists all scenes in the examples folder)")
    gr.Markdown("Click any thumbnail to load the scene for reconstruction.")

    # Get scene information
    scenes = get_scene_info("examples")

    # Create thumbnail grid (4 columns, N rows)
    if scenes:
        for i in range(0, len(scenes), 4):  # Process 4 scenes per row
            with gr.Row():
                for j in range(4):
                    scene_idx = i + j
                    if scene_idx < len(scenes):
                        scene = scenes[scene_idx]
                        with gr.Column(scale=1, elem_classes=["clickable-thumbnail"]):
                            # Clickable thumbnail
                            scene_img = gr.Image(
                                value=scene["thumbnail"],
                                height=150,
                                interactive=False,
                                show_label=False,
                                elem_id=f"scene_thumb_{scene['name']}",
                                sources=[],
                            )

                            # Scene name and image count as text below thumbnail
                            gr.Markdown(
                                f"**{scene['name']}** \n {scene['num_images']} images",
                                elem_classes=["scene-info"],
                            )

                            # Connect thumbnail click to load scene
                            scene_img.select(
                                fn=lambda name=scene["name"]: load_example_scene(name),
                                outputs=[
                                    reconstruction_output,
                                    target_dir_output,
                                    image_gallery,
                                    log_output,
                                ],
                            )
                    else:
                        # Empty column to maintain grid structure
                        with gr.Column(scale=1):
                            pass

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            frame_filter,
            show_cam,
            filter_black_bg,
            filter_white_bg,
            conf_thres,
            apply_mask_checkbox,
            show_mesh,
        ],
        outputs=[
            reconstruction_output,
            log_output,
            frame_filter,
            processed_data_state,
            depth_map,
            normal_map,
            measure_image,
            measure_text,
            depth_view_selector,
            normal_view_selector,
            measure_view_selector,
        ],
    ).then(
        fn=lambda: "False",
        inputs=[],
        outputs=[is_example],  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            conf_thres,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            conf_thres,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
        ],
        [reconstruction_output, log_output],
    )
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            conf_thres,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
        ],
        [reconstruction_output, log_output],
    )
    filter_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            conf_thres,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
        ],
        [reconstruction_output, log_output],
    ).then(
        fn=update_all_views_on_filter_change,
        inputs=[
            target_dir_output,
            filter_black_bg,
            filter_white_bg,
            processed_data_state,
            depth_view_selector,
            normal_view_selector,
            measure_view_selector,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
            measure_image,
            measure_points_state,
        ],
    )
    filter_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            conf_thres,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
        ],
        [reconstruction_output, log_output],
    ).then(
        fn=update_all_views_on_filter_change,
        inputs=[
            target_dir_output,
            filter_black_bg,
            filter_white_bg,
            processed_data_state,
            depth_view_selector,
            normal_view_selector,
            measure_view_selector,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
            measure_image,
            measure_points_state,
        ],
    )
    show_mesh.change(
        update_visualization,
        [
            target_dir_output,
            frame_filter,
            show_cam,
            is_example,
            filter_black_bg,
            filter_white_bg,
            show_mesh,
            conf_thres,
        ],
        [reconstruction_output, log_output],
    )
    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, s_time_interval],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, s_time_interval],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    # -------------------------------------------------------------------------
    # Measure tab functionality
    # -------------------------------------------------------------------------
    measure_image.select(
        fn=measure,
        inputs=[processed_data_state, measure_points_state, measure_view_selector],
        outputs=[measure_image, measure_points_state, measure_text],
    )

    # -------------------------------------------------------------------------
    # Navigation functionality for Depth, Normal, and Measure tabs
    # -------------------------------------------------------------------------

    # Depth tab navigation
    prev_depth_btn.click(
        fn=lambda processed_data, current_selector: navigate_depth_view(
            processed_data, current_selector, -1
        ),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_view_selector, depth_map],
    )

    next_depth_btn.click(
        fn=lambda processed_data, current_selector: navigate_depth_view(
            processed_data, current_selector, 1
        ),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_view_selector, depth_map],
    )

    depth_view_selector.change(
        fn=lambda processed_data, selector_value: (
            update_depth_view(
                processed_data,
                int(selector_value.split()[1]) - 1,
            )
            if selector_value
            else None
        ),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_map],
    )

    # Normal tab navigation
    prev_normal_btn.click(
        fn=lambda processed_data, current_selector: navigate_normal_view(
            processed_data, current_selector, -1
        ),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_view_selector, normal_map],
    )

    next_normal_btn.click(
        fn=lambda processed_data, current_selector: navigate_normal_view(
            processed_data, current_selector, 1
        ),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_view_selector, normal_map],
    )

    normal_view_selector.change(
        fn=lambda processed_data, selector_value: (
            update_normal_view(
                processed_data,
                int(selector_value.split()[1]) - 1,
            )
            if selector_value
            else None
        ),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_map],
    )

    # Measure tab navigation
    prev_measure_btn.click(
        fn=lambda processed_data, current_selector: navigate_measure_view(
            processed_data, current_selector, -1
        ),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_view_selector, measure_image, measure_points_state],
    )

    next_measure_btn.click(
        fn=lambda processed_data, current_selector: navigate_measure_view(
            processed_data, current_selector, 1
        ),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_view_selector, measure_image, measure_points_state],
    )

    measure_view_selector.change(
        fn=lambda processed_data, selector_value: (
            update_measure_view(processed_data, int(selector_value.split()[1]) - 1)
            if selector_value
            else (None, [])
        ),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_image, measure_points_state],
    )

    # -------------------------------------------------------------------------
    # Acknowledgement section
    # -------------------------------------------------------------------------
    gr.HTML(get_acknowledgements_html())

    demo.queue(max_size=20).launch(show_error=True, share=True, ssr_mode=False)
