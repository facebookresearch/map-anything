# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for loading, converting, and manipulating images.

This module provides functions for:
- Converting between different image formats and representations
- Resizing and cropping images to specific resolutions
- Loading and normalizing images for model input
- Handling various image file formats including HEIF/HEIC when available
"""

import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.geometry import recover_pinhole_intrinsics_from_ray_directions
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
    504: {
        1.000: (504, 504),  # 1:1
        1.333: (504, 378),  # 4:3
        1.565: (504, 322),  # 3:2
        1.800: (504, 280),  # 16:9
        2.118: (504, 238),  # 2:1
        3.273: (504, 154),  # 3.2:1
        0.750: (378, 504),  # 3:4
        0.639: (322, 504),  # 2:3
        0.556: (280, 504),  # 9:16
        0.472: (238, 504),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
    504: sorted(RESOLUTION_MAPPINGS[504].keys()),
}


def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]


def rgb(ftensor, norm_type, true_shape=None):
    """
    Convert normalized image tensor to RGB image for visualization.

    Args:
        ftensor (torch.Tensor or numpy.ndarray or list): Image tensor or list of image tensors
        norm_type (str): Normalization type, see UniCeption IMAGE_NORMALIZATION_DICT keys or use "identity"
        true_shape (tuple, optional): If provided, the image will be cropped to this shape (H, W)

    Returns:
        numpy.ndarray: RGB image with values in range [0, 1]
    """
    if isinstance(ftensor, list):
        return [rgb(x, norm_type, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            mean = img_norm.mean.numpy()
            std = img_norm.std.numpy()
        elif norm_type == "identity":
            mean = 0.0
            std = 1.0
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available types: identity or {IMAGE_NORMALIZATION_DICT.keys()}"
            )
        img = ftensor * std + mean
    return img.clip(min=0, max=1)


def load_images(
    folder_or_list,
    resize_mode="fixed_mapping",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    verbose=False,
    bayer_format=False,
    resolution_set=518,
    stride=1,
):
    """
    Open and convert all images in a list or folder to proper input format for model

    Args:
        folder_or_list (str or list): Path to folder or list of image paths.
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): Required for "longest_side", "square", and "fixed_size" modes.
                                      - For "longest_side" and "square": int value for resize dimension
                                      - For "fixed_size": tuple of (width, height)
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        bayer_format (bool, optional): If True, read images in Bayer format. Defaults to False.
        resolution_set (int, optional): Resolution set to use for "fixed_mapping" mode (518 or 512). Defaults to 518.
        stride (int, optional): Load every nth image from the input. stride=1 loads all images, stride=2 loads every 2nd image, etc. Defaults to 1.

    Returns:
        list: List of dictionaries containing image data and metadata
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
        raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    # Get list of image paths
    if isinstance(folder_or_list, str):
        # If folder_or_list is a string, assume it's a path to a folder
        if verbose:
            print(f"Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        # If folder_or_list is a list, assume it's a list of image paths
        if verbose:
            print(f"Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        # If folder_or_list is neither a string nor a list, raise an error
        raise ValueError(f"Bad {folder_or_list=} ({type(folder_or_list)})")

    # Define supported image extensions
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    # First pass: Load all images and collect aspect ratios
    loaded_images = []
    aspect_ratios = []
    for i, path in enumerate(folder_content):
        # Skip images based on stride
        if i % stride != 0:
            continue

        # Check if the file has a supported image extension
        if not path.lower().endswith(supported_images_extensions):
            continue

        try:
            if bayer_format:
                # If bayer_format is True, read the image in Bayer format
                color_bayer = cv2.imread(os.path.join(root, path), cv2.IMREAD_UNCHANGED)
                color = cv2.cvtColor(color_bayer, cv2.COLOR_BAYER_RG2BGR)
                img = PIL.Image.fromarray(color)
                img = exif_transpose(img).convert("RGB")
            else:
                # Otherwise, read the image normally
                img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert(
                    "RGB"
                )

            W1, H1 = img.size
            aspect_ratios.append(W1 / H1)
            loaded_images.append((path, img, W1, H1))

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {path}: {e}")
            continue

    # Check if any images were loaded
    if not loaded_images:
        raise ValueError("No valid images found")

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)
    elif resize_mode == "square":
        target_size = (
            round((size // patch_size)) * patch_size,
            round((size // patch_size)) * patch_size,
        )
    elif resize_mode == "longest_side":
        # Use average aspect ratio to determine size for all images
        # Longest side should be the input size
        if average_aspect_ratio >= 1:  # Landscape or square
            # Width is the longest side
            target_size = (
                size,
                round((size // patch_size) / average_aspect_ratio) * patch_size,
            )
        else:  # Portrait
            # Height is the longest side
            target_size = (
                round((size // patch_size) * average_aspect_ratio) * patch_size,
                size,
            )
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all images"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Second pass: Resize all images to the same target size
    imgs = []
    for path, img, W1, H1 in loaded_images:
        # Resize and crop the image to the target size
        img = crop_resize_if_necessary(img, resolution=target_size)[0]

        # Normalize image and add it to the list
        W2, H2 = img.size
        if verbose:
            print(f" - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
            )
        )

    assert imgs, "No images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")

    return imgs


def preprocess_inputs(
    input_views,
    resize_mode="fixed_mapping",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    resolution_set=518,
    verbose=False,
):
    """
    Preprocess input_views by determining optimal aspect ratio and resizing all images and multi-modal inputs.

    Similar to load_images function, this function:
    (a) Determines the optimal aspect ratio from all input images
    (b) Resizes all images and multi-modal inputs using crop_resize_if_necessary
    (c) Normalizes images according to the specified normalization type

    Args:
        input_views (list): List of dictionaries containing view data. Each view can contain:
            - img: Image tensor (H, W, 3) - [0, 255] or PIL Image
            - intrinsics: Camera intrinsics (3, 3)
            - depth_z: Depth maps (H, W)
            - ray_directions: Ray directions (H, W, 3)
            - camera_poses: Camera poses (4, 4) or tuple of (quats, trans) - not resized
            - is_metric_scale: Boolean value - not resized
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): Required for "longest_side", "square", and "fixed_size" modes.
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        resolution_set (int, optional): Resolution set to use for "fixed_mapping" mode (518 or 512). Defaults to 518.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        list: List of processed view dictionaries with resized images and multi-modal inputs
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
        raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    if not input_views:
        raise ValueError("input_views cannot be empty")

    # First pass: Extract all images and collect aspect ratios
    aspect_ratios = []
    for view_idx, view in enumerate(input_views):
        if "img" not in view:
            if verbose:
                print(
                    f"Warning: View {view_idx} has no 'img' key, skipping for aspect ratio calculation"
                )
            continue

        img = view["img"]

        # Handle different image formats (no batch dimension expected)
        if isinstance(img, torch.Tensor):
            # Tensor format: (H, W, 3) - channel last
            if img.ndim == 3 and img.shape[2] == 3:
                H, W = img.shape[0], img.shape[1]
            else:
                raise ValueError(
                    f"Expected tensor shape (H, W, 3) for img in view {view_idx}, got {img.shape}"
                )
        elif isinstance(img, PIL.Image.Image):
            W, H = img.size
        elif isinstance(img, np.ndarray):
            # Array format: (H, W, 3) - channel last
            if img.ndim == 3 and img.shape[2] == 3:
                H, W = img.shape[0], img.shape[1]
            else:
                raise ValueError(
                    f"Expected array shape (H, W, 3) for img in view {view_idx}, got {img.shape}"
                )
        else:
            raise ValueError(f"Unsupported image type in view {view_idx}: {type(img)}")

        aspect_ratios.append(W / H)

    if not aspect_ratios:
        raise ValueError("No valid images found in input_views")

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)
    elif resize_mode == "square":
        target_size = (
            round((size // patch_size)) * patch_size,
            round((size // patch_size)) * patch_size,
        )
    elif resize_mode == "longest_side":
        # Use average aspect ratio to determine size for all images
        if average_aspect_ratio >= 1:  # Landscape or square
            target_size = (
                size,
                round((size // patch_size) / average_aspect_ratio) * patch_size,
            )
        else:  # Portrait
            target_size = (
                round((size // patch_size) * average_aspect_ratio) * patch_size,
                size,
            )
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all views"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Helper function to convert tensor/array to PIL Image
    def to_pil_image(img, view_idx):
        """Convert tensor or array to PIL Image for processing."""
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL Image for processing - expect (H, W, 3)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Expected tensor shape (H, W, 3) for img in view {view_idx}, got {img.shape}"
                )
            # Only multiply with 255 if the image range is within [0, 1]
            if img.max() <= 1.0:
                img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            else:
                img = img.clamp(0, 255).byte().cpu().numpy()
            return PIL.Image.fromarray(img)
        elif isinstance(img, np.ndarray):
            # Expect (H, W, 3) format
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Expected array shape (H, W, 3) for img in view {view_idx}, got {img.shape}"
                )
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            return PIL.Image.fromarray(img)
        elif isinstance(img, PIL.Image.Image):
            return img
        else:
            raise ValueError(f"Unsupported image type in view {view_idx}: {type(img)}")

    # Helper function to convert tensor to numpy array
    def to_numpy(data, expected_shape, name, view_idx):
        """Convert tensor to numpy array and validate shape."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not isinstance(data, np.ndarray):
            raise ValueError(
                f"Expected tensor or array for {name} in view {view_idx}, got {type(data)}"
            )

        if data.shape != expected_shape and expected_shape is not None:
            raise ValueError(
                f"Expected shape {expected_shape} for {name} in view {view_idx}, got {data.shape}"
            )

        return data

    # Second pass: Resize all images and multi-modal inputs
    processed_views = []
    for view_idx, view in enumerate(input_views):
        # Convert image to PIL format
        if "img" not in view:
            raise ValueError(f"View {view_idx} missing required 'img' key")

        img = to_pil_image(view["img"], view_idx)

        # Prepare inputs for crop_resize_if_necessary
        depthmap = None
        intrinsics = None

        # Handle depth_z
        if "depth_z" in view:
            depthmap = to_numpy(view["depth_z"], None, "depth_z", view_idx)
            if depthmap.ndim != 2:
                raise ValueError(
                    f"Expected shape (H, W) for depth_z in view {view_idx}, got {depthmap.shape}"
                )

        # Enforce that only one of intrinsics and ray_directions is provided
        has_intrinsics = "intrinsics" in view
        has_ray_directions = "ray_directions" in view

        if has_intrinsics and has_ray_directions:
            raise ValueError(
                f"View {view_idx} cannot have both 'intrinsics' and 'ray_directions'. "
                "Please provide only one as they are redundant (ray_directions can be used to recover intrinsics)."
            )

        # Handle intrinsics
        if has_intrinsics:
            intrinsics = to_numpy(view["intrinsics"], (3, 3), "intrinsics", view_idx)

        # Handle ray_directions by recovering intrinsics from them
        if has_ray_directions:
            ray_dirs = to_numpy(
                view["ray_directions"], None, "ray_directions", view_idx
            )
            if ray_dirs.ndim != 3 or ray_dirs.shape[2] != 3:
                raise ValueError(
                    f"Expected shape (H, W, 3) for ray_directions in view {view_idx}, got {ray_dirs.shape}"
                )

            # Convert ray directions to torch tensor for the geometry function
            ray_dirs_torch = torch.from_numpy(ray_dirs)

            # Recover intrinsics from ray directions
            recovered_intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                ray_dirs_torch
            )
            recovered_intrinsics = recovered_intrinsics.cpu().numpy()
            intrinsics = recovered_intrinsics

        # Process all inputs with a single call to crop_resize_if_necessary
        results = crop_resize_if_necessary(
            image=img,
            resolution=target_size,
            depthmap=depthmap,
            intrinsics=intrinsics,
        )

        # Unpack results based on what was provided
        processed_view = {}
        result_idx = 0

        # Image is always first - normalize it after resizing
        resized_img = results[result_idx]
        processed_view["img"] = ImgNorm(resized_img)[
            None
        ]  # Add batch dimension like load_images
        processed_view["data_norm_type"] = [norm_type]  # Add normalization type
        result_idx += 1

        # Depth is next if provided - add batch dimension
        if depthmap is not None:
            processed_view["depth_z"] = torch.from_numpy(results[result_idx])[None]
            result_idx += 1

        # Intrinsics is next if provided - add batch dimension
        if intrinsics is not None:
            processed_view["intrinsics"] = torch.from_numpy(results[result_idx])[None]
            result_idx += 1

        # Handle camera_poses with batch dimension if present
        if "camera_poses" in view:
            camera_poses = view["camera_poses"]
            if isinstance(camera_poses, tuple):
                # Tuple format (quats, trans) - add batch dimension to both components
                quats, trans = camera_poses
                if isinstance(quats, torch.Tensor):
                    quats_batched = quats[None]
                elif isinstance(quats, np.ndarray):
                    quats_batched = torch.from_numpy(quats)[None]
                else:
                    quats_batched = torch.tensor(quats)[None]
                if isinstance(trans, torch.Tensor):
                    trans_batched = trans[None]
                elif isinstance(trans, np.ndarray):
                    trans_batched = torch.from_numpy(trans)[None]
                else:
                    trans_batched = torch.tensor(trans)[None]
                processed_view["camera_poses"] = (quats_batched, trans_batched)
            else:
                # Matrix format - add batch dimension
                if isinstance(camera_poses, torch.Tensor):
                    processed_view["camera_poses"] = camera_poses[None]
                elif isinstance(camera_poses, np.ndarray):
                    processed_view["camera_poses"] = torch.from_numpy(camera_poses)[
                        None
                    ]
                else:
                    raise ValueError(
                        f"Unsupported camera_poses format: {type(camera_poses)}. Expected tuple (quats, trans) or matrix (tensor/array)."
                    )

        # Copy over any other keys that don't need resizing or batch dimensions
        for key, value in view.items():
            if key not in [
                "img",
                "depth_z",
                "intrinsics",
                "ray_directions",
                "camera_poses",
            ]:
                processed_view[key] = value

        processed_views.append(processed_view)

        if verbose:
            print(f"Processed view {view_idx} with keys: {list(processed_view.keys())}")

    if verbose:
        print(f"Successfully processed {len(processed_views)} views")

    return processed_views
