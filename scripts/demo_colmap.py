# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to get MapAnything outputs in COLMAP format. Optionally can also run BA on outputs.

Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)
"""

import argparse
import copy
import glob
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torchvision import transforms as tvf

from morphcloud.models import MapAnything
from morphcloud.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from morphcloud.third_party.track_predict import predict_tracks
from morphcloud.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from morphcloud.utils.image import rgb
from morphcloud.utils.misc import seed_everything
from morphcloud.utils.viz import predictions_to_glb
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="MapAnything COLMAP Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing the scene images",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=0.0,
        help="Confidence threshold value for depth filtering (used only without BA)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save dense reconstruction (without BA) as GLB file",
    )
    parser.add_argument(
        "--use_ba", action="store_true", default=False, help="Use BA for reconstruction"
    )
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Maximum reprojection error for reconstruction",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Use shared camera for all images",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="Camera type for reconstruction",
    )
    parser.add_argument(
        "--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks"
    )
    parser.add_argument(
        "--query_frame_num", type=int, default=8, help="Number of frames to query"
    )
    parser.add_argument(
        "--max_query_pts", type=int, default=4096, help="Maximum number of query points"
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Use fine tracking (slower but more accurate)",
    )
    return parser.parse_args()


def load_and_preprocess_images_square(
    image_path_list, target_size=1024, data_norm_type=None
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        data_norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to None (no normalization).

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty or if an invalid data_norm_type is provided
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive

    # Set up normalization based on data_norm_type
    if data_norm_type is None:
        # No normalization, just convert to tensor
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        # Use the specified normalization
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize(
            (target_size, target_size), Image.Resampling.BICUBIC
        )

        # Convert to tensor and apply normalization
        img_tensor = img_transform(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(
        true_indices, size=max_trues, replace=False
    )  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def run_morphcloud(
    model,
    images,
    dtype,
    resolution=518,
    image_normalization_type="dinov2",
    memory_efficient_inference=False,
):
    # Images: [V, 3, H, W]
    # Check image shape
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Hard-coded to use 518 for MapAnything
    images = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    # Run inference
    views = []
    for view_idx in range(images.shape[0]):
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [image_normalization_type],
        }
        views.append(view)
    predictions = model.infer(
        views, memory_efficient_inference=memory_efficient_inference
    )

    # Process predictions
    (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for pred in predictions:
        # Compute 3D points from depth, intrinsics, and camera pose
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        pts3d, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Extract mask from predictions and combine with valid depth mask
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask

        # Convert tensors to numpy arrays
        extrinsic = (
            closed_form_pose_inverse(pred["camera_poses"])[0].cpu().numpy()
        )  # c2w -> w2c
        intrinsic = intrinsics_torch.cpu().numpy()
        depth_map = depthmap_torch.cpu().numpy()
        depth_conf = pred["conf"][0].cpu().numpy()
        pts3d = pts3d.cpu().numpy()
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # Denormalized image

        # Collect results
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_pts3d.append(pts3d)
        all_img_no_norm.append(img_no_norm)
        all_masks.append(mask)

    # Stack results into arrays
    all_extrinsics = np.stack(all_extrinsics)
    all_intrinsics = np.stack(all_intrinsics)
    all_depth_maps = np.stack(all_depth_maps)
    all_depth_confs = np.stack(all_depth_confs)
    all_pts3d = np.stack(all_pts3d)
    all_img_no_norm = np.stack(all_img_no_norm)
    all_masks = np.stack(all_masks)

    return (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    )


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Set device and dtype
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Init model
    print("Loading MapAnything model from huggingface ...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running MapAnything with 518
    morphcloud_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(
        image_path_list, img_load_resolution, model.encoder.data_norm_type
    )
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run MapAnything to estimate camera and depth
    # Run with 518 x 518 images
    extrinsic, intrinsic, depth_map, depth_conf, points_3d, img_no_norm, masks = (
        run_morphcloud(
            model,
            images,
            dtype,
            morphcloud_fixed_resolution,
            model.encoder.data_norm_type,
            memory_efficient_inference=args.memory_efficient_inference,
        )
    )

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    if args.save_glb:
        for i in range(img_no_norm.shape[0]):
            # Use the already denormalized images from predictions
            images_list.append(img_no_norm[i])

            # Add world points and masks from predictions
            world_points_list.append(points_3d[i])
            masks_list.append(masks[i])  # Use masks from predictions

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / morphcloud_fixed_resolution
        shared_camera = args.shared_camera

        with torch.amp.autocast("cuda", dtype=dtype):
            # Predicting Tracks
            # Uses VGGSfM tracker
            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = (
                predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )
            )

            torch.cuda.empty_cache()

        # Rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # Init pycolmap reconstruction
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = (
            False  # in the feedforward manner, we do not support shared camera
        )
        camera_type = (
            "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera
        )

        image_size = np.array(
            [morphcloud_fixed_resolution, morphcloud_fixed_resolution]
        )
        num_frames, height, width, _ = points_3d.shape

        # Denormalize images before computing RGB values
        points_rgb_images = F.interpolate(
            images,
            size=(morphcloud_fixed_resolution, morphcloud_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Convert normalized images back to RGB [0,1] range using the rgb function
        points_rgb_list = []
        for i in range(points_rgb_images.shape[0]):
            # rgb function expects single image tensor and returns numpy array in [0,1] range
            rgb_img = rgb(points_rgb_images[i], model.encoder.data_norm_type)
            points_rgb_list.append(rgb_img)

        # Stack and convert to uint8
        points_rgb = np.stack(points_rgb_list)  # Shape: (N, H, W, 3)
        points_rgb = (points_rgb * 255).astype(np.uint8)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # At most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = morphcloud_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(args.scene_dir, "sparse/points.ply")
    )

    # Export GLB if requested
    if args.save_glb:
        glb_output_path = os.path.join(args.scene_dir, "dense_mesh.glb")
        print(f"Saving GLB file to: {glb_output_path}")

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)

        # Save GLB file
        scene_3d.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded & resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # No need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
