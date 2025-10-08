# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Base class for MapAnything datasets.
"""

from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation

from mapanything.datasets.base.easy_dataset import EasyDataset
from mapanything.utils.cropping import (
    bbox_from_intrinsics_in_out,
    camera_matrix_of_crop,
    crop_image_and_other_optional_info,
    rescale_image_and_other_optional_info,
)
from mapanything.utils.geometry import (
    depthmap_to_camera_coordinates,
    get_absolute_pointmaps_and_rays_info,
)
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


class BaseDataset(EasyDataset):
    """
    Define all basic options.

    Usage:
        class MyDataset(BaseDataset):
            def _get_views(self, idx):
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        num_views: int,
        variable_num_views: bool = False,
        split: str = None,
        covisibility_thres: float = None,
        resolution: Union[int, Tuple[int, int], List[Tuple[int, int]]] = None,
        principal_point_centered: bool = False,
        transform: str = None,
        data_norm_type: str = None,
        aug_crop: int = 0,
        seed: int = None,
        max_num_retries: int = 5,
    ):
        """
        PyTorch dataset for multi-view images sampled from scenes, where the images form a single connected component.

        Args:
            num_views (int): Number of views.
            variable_num_views (bool): If True, the number of views can vary from batch to batch. The maximum number of views is num_views and minimum is 2.
                                       On by default for N-view train dataloader (hydra config).
            split (str): 'train', 'val', 'test', etc.
            covisibility_thres (float): Covisibility (%) threshold to determine if another image is a neighbor or not
            resolution (int or tuple or list of tuples): Resolution of the images
            principal_point_centered (bool): If True, the principal point is centered in the image.
            transform (str): Transform to apply to the images. Options:
            - 'colorjitter+grayscale+gaublur':
                tvf.Compose([
                    tvf.RandomApply([tvf.ColorJittter(0.3, 0.4, 0.2, 0.1)], p=0.75),
                    tvf.RandomGrayscale(p=0.05),
                    tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                ]) after ImgNorm
            - 'colorjitter': tvf.ColorJittter(0.5, 0.5, 0.5, 0.1) after ImgNorm
            - 'imgnorm': ImgNorm only
            data_norm_type (str): Image normalization type.
                                  For options, see UniCeption image normalization dict.
            aug_crop (int): Augment crop. If int greater than 0, indicates the number of pixels to increase in target resolution.
            seed (int): Seed for the random number generator.
            max_num_retries (int): Maximum number of retries for loading a different sample from the dataset, if provided idx fails.
        """
        self.num_views = num_views
        self.variable_num_views = variable_num_views
        self.num_views_min = 2
        self.split = split
        self.covisibility_thres = covisibility_thres
        self._set_resolutions(resolution)
        self.principal_point_centered = principal_point_centered

        # Update the number of views if necessary and make it a list if variable_num_views is True
        if self.variable_num_views and self.num_views > self.num_views_min:
            self.num_views = list(range(self.num_views_min, self.num_views + 1))

        # Initialize the image normalization type
        if data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            self.data_norm_type = data_norm_type
            image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
            ImgNorm = tvf.Compose(
                [
                    tvf.ToTensor(),
                    tvf.Normalize(mean=image_norm.mean, std=image_norm.std),
                ]
            )
        elif data_norm_type == "identity":
            self.data_norm_type = data_norm_type
            ImgNorm = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(
                f"Unknown data_norm_type: {data_norm_type}. Available options: identity or {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )

        # Initialize torchvision transforms
        if transform == "imgnorm":
            self.transform = ImgNorm
        elif transform == "colorjitter":
            self.transform = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
        elif transform == "colorjitter+grayscale+gaublur":
            self.transform = tvf.Compose(
                [
                    tvf.RandomApply([tvf.ColorJitter(0.3, 0.4, 0.2, 0.1)], p=0.75),
                    tvf.RandomGrayscale(p=0.05),
                    tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                    ImgNorm,
                ]
            )
        else:
            raise ValueError(
                'Unknown transform. Available options: "imgnorm", "colorjitter", "colorjitter+grayscale+gaublur"'
            )

        # Initialize the augmentation parameters
        self.aug_crop = aug_crop

        # Initialize the seed for the random number generator
        self.seed = seed
        self._seed_offset = 0

        # Initialize the maximum number of retries for loading a different sample from the dataset, if the first idx fails
        self.max_num_retries = max_num_retries

        # Initialize the dataset type flags
        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this
        self.is_synthetic = False  # by default a dataset is not synthetic, subclasses can overwrite this

    def _load_data(self):
        self.scenes = []
        self.num_of_scenes = len(self.scenes)

    def __len__(self):
        "Length of the dataset is determined by the number of scenes in the dataset split"
        return self.num_of_scenes

    def get_stats(self):
        "Get the number of scenes in the dataset split"
        return f"{self.num_of_scenes} scenes"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=}
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace("self.", "")
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, num_views_to_sample, resolution):
        raise NotImplementedError()

    def _set_seed_offset(self, idx):
        """
        Set the seed offset. This is directly added to self.seed when setting the random seed.
        """
        self._seed_offset = idx

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if isinstance(resolutions, int):
            resolutions = [resolutions]
        elif isinstance(resolutions, tuple):
            resolutions = [resolutions]
        elif isinstance(resolutions, list):
            assert all(isinstance(res, tuple) for res in resolutions), (
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )
        else:
            raise ValueError(
                f"Bad type for {resolutions=}, should be int or tuple of ints or list of tuples of ints"
            )

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), (
                f"Bad type for {width=} {type(width)=}, should be int"
            )
            assert isinstance(height, int), (
                f"Bad type for {height=} {type(height)=}, should be int"
            )
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self,
        image,
        resolution,
        depthmap,
        intrinsics,
        additional_quantities=None,
    ):
        """
        Process an image by downsampling and cropping as needed to match the target resolution.

        This method performs the following operations:
        1. Converts the image to PIL.Image if necessary
        2. Crops the image centered on the principal point if requested
        3. Downsamples the image using high-quality Lanczos filtering
        4. Performs final cropping to match the target resolution

        Args:
            image (numpy.ndarray or PIL.Image.Image): Input image to be processed
            resolution (tuple): Target resolution as (width, height)
            depthmap (numpy.ndarray): Depth map corresponding to the image
            intrinsics (numpy.ndarray): Camera intrinsics matrix (3x3)
            additional_quantities (dict, optional): Additional image-related data to be processed
                                                   alongside the main image with nearest interpolation. Defaults to None.

        Returns:
            tuple: Processed image, depthmap, and updated intrinsics matrix.
                  If additional_quantities is provided, it returns those as well.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Cropping centered on the principal point if necessary
        if self.principal_point_centered:
            W, H = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                # Skip centered cropping if principal point is outside image bounds
                pass
            else:
                min_margin_x = min(cx, W - cx)
                min_margin_y = min(cy, H - cy)
                left, top = cx - min_margin_x, cy - min_margin_y
                right, bottom = cx + min_margin_x, cy + min_margin_y
                crop_bbox = (left, top, right, bottom)
                # Only perform the centered crop if the crop_bbox is larger than the target resolution
                crop_width = right - left
                crop_height = bottom - top
                if crop_width > resolution[0] and crop_height > resolution[1]:
                    image, depthmap, intrinsics, additional_quantities = (
                        crop_image_and_other_optional_info(
                            image=image,
                            crop_bbox=crop_bbox,
                            depthmap=depthmap,
                            camera_intrinsics=intrinsics,
                            additional_quantities=additional_quantities,
                        )
                    )

        # Get the target resolution for re-scaling
        target_rescale_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_rescale_resolution += self._rng.integers(0, self.aug_crop)

        # High-quality Lanczos down-scaling if necessary
        image, depthmap, intrinsics, additional_quantities = (
            rescale_image_and_other_optional_info(
                image=image,
                output_resolution=target_rescale_resolution,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities_to_be_resized_with_nearest=additional_quantities,
            )
        )

        # Actual cropping (if necessary)
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
        image, depthmap, new_intrinsics, additional_quantities = (
            crop_image_and_other_optional_info(
                image=image,
                crop_bbox=crop_bbox,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities=additional_quantities,
            )
        )

        # Return the output
        if additional_quantities is not None:
            return image, depthmap, new_intrinsics, additional_quantities
        else:
            return image, depthmap, new_intrinsics

    def _random_walk_sampling(
        self,
        scene_pairwise_covisibility,
        num_of_samples,
        max_retries=4,
        use_bidirectional_covis=True,
    ):
        """
        Randomly samples S indices from an N x N covisibility matrix by forming adjacency edges such that the resulting subgraph (given by the indices) is connected.
        If the current node has no new unvisited neighbors, backtracking occurs.
        Retries with different starting indices if the desired number of samples is not reached, excluding previously visited components.

        Args:
            scene_pairwise_covisibility : np.ndarray (mmap)
                N x N covisibility matrix for the scene, where N is the number of views in the scene.
            num_of_samples : int
                The desired number of nodes to sample (num_of_samples < N).
            max_retries : int
                The maximum number of retries with different starting indices.
            use_bidirectional_covis : bool
                Whether to compute bidirectional covisibility by averaging row and column values.
                If False, uses only row access (faster for large memory-mapped arrays).
                Defaults to True.

        Returns:
            np.ndarray
                An array of sampled indices forming a connected subgraph.
        """
        excluded_nodes = set()
        best_walk = []  # To keep track of the best walk found
        for _ in range(max_retries):
            visited = set()
            walk = []  # List to store the random walk sampling order
            stack = []  # Stack for backtracking

            # Choose a random starting index that is not in the excluded set
            all_nodes = set(range(len(scene_pairwise_covisibility)))
            available_nodes = list(all_nodes - excluded_nodes)
            if not available_nodes:
                break  # No more nodes to try
            start = self._rng.choice(available_nodes)
            walk.append(start)
            visited.add(start)
            stack.append(start)

            # Continue until we have sampled S indices or all expandable nodes are exhausted
            while len(walk) < num_of_samples and stack:
                current = stack[-1]
                # Get the pairwise covisibility for the current node
                if use_bidirectional_covis:
                    # Use bidirectional covisibility (slower for large memory-mapped arrays)
                    pairwise_covisibility = (
                        scene_pairwise_covisibility[current, :]
                        + scene_pairwise_covisibility[:, current].T
                    ) / 2
                else:
                    # Use only row access (faster for large memory-mapped arrays)
                    pairwise_covisibility = scene_pairwise_covisibility[current, :]
                # Normalize the covisibility using self covisibility
                pairwise_covisibility = pairwise_covisibility / (
                    pairwise_covisibility[current] + 1e-8
                )
                # Assign overlap score of zero to self-pairs
                pairwise_covisibility[current] = 0
                # Threshold the covisibility to get adjacency list for the current node
                adjacency_list_for_current = (
                    pairwise_covisibility > self.covisibility_thres
                ).astype(int)
                adjacency_list_for_current = np.flatnonzero(adjacency_list_for_current)
                # Get all unvisited neighbors
                candidates = [
                    idx for idx in adjacency_list_for_current if idx not in visited
                ]  # Remove visited nodes
                if candidates:
                    # Randomly select one of the unvisited overlapping neighbors
                    next_node = self._rng.choice(candidates)
                    walk.append(next_node)
                    visited.add(next_node)
                    stack.append(next_node)
                else:
                    # If no unvisited neighbor is available, backtrack
                    stack.pop()

            # Update the best walk if the current walk is larger
            if len(walk) > len(best_walk):
                best_walk = walk

            # If we have enough samples, return the result
            if len(walk) >= num_of_samples:
                return np.array(walk)

            # Add all visited nodes to the excluded set
            excluded_nodes.update(visited)

        # If all retries are exhausted and we still don't have enough samples, return the best walk found
        return np.array(best_walk)

    def _sample_view_indices(
        self,
        num_views_to_sample,
        num_views_in_scene,
        scene_pairwise_covisibility,
        use_bidirectional_covis=True,
    ):
        """
        Sample view indices from a scene based on the adjacency list and the number of views to sample.

        Args:
            num_views_to_sample (int): Number of views to sample.
            num_views_in_scene (int): Total number of views available in the scene.
            scene_pairwise_covisibility (np.ndarray): N x N covisibility matrix for the scene, where N is the number of views in the scene.
            use_bidirectional_covis (bool): Whether to compute bidirectional covisibility by averaging row and column values.
                If False, uses only row access (faster for large memory-mapped arrays).

        Returns:
            numpy.ndarray: Array of sampled view indices.
        """
        if num_views_to_sample == num_views_in_scene:
            # Select all views in the scene
            view_indices = self._rng.permutation(num_views_in_scene)
        elif num_views_to_sample > num_views_in_scene:
            # Select all views in the scene and repeat them to get the desired number of views
            view_indices = self._rng.choice(
                num_views_in_scene, size=num_views_to_sample, replace=True
            )
        else:
            # Select a subset of single component connected views in the scene using random walk sampling
            view_indices = self._random_walk_sampling(
                scene_pairwise_covisibility,
                num_views_to_sample,
                use_bidirectional_covis=use_bidirectional_covis,
            )
            # If the required num of views can't be obtained even with 4 retries, repeat existing indices to get the desired number of views
            if len(view_indices) < num_views_to_sample:
                view_indices = self._rng.choice(
                    view_indices, size=num_views_to_sample, replace=True
                )

        return view_indices

    def _getitem_fn(self, idx):
        if isinstance(idx, tuple):
            # The idx is a tuple if specifying the aspect-ratio or/and the number of views
            if isinstance(self.num_views, int):
                idx, ar_idx = idx
            else:
                idx, ar_idx, num_views_to_sample_idx = idx
        else:
            assert len(self._resolutions) == 1
            assert isinstance(self.num_views, int)
            ar_idx = 0

        # Setup the rng
        if self.seed:  # reseed for each _getitem_fn
            # Leads to deterministic sampling where repeating self.seed and self._seed_offset yields the same multi-view set again
            # Scenes will be repeated if size of dataset is artificially increased using "N @" or "N *"
            # When scenes are repeated, self._seed_offset is increased to ensure new multi-view sets
            # This is useful for evaluation if the number of dataset scenes is < N, yet we want unique multi-view sets each iter
            self._rng = np.random.default_rng(seed=self.seed + self._seed_offset + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # Get the views for the given index and check that the number of views is correct
        resolution = self._resolutions[ar_idx]
        if isinstance(self.num_views, int):
            num_views_to_sample = self.num_views
        else:
            num_views_to_sample = self.num_views[num_views_to_sample_idx]
        views = self._get_views(idx, num_views_to_sample, resolution)
        if isinstance(self.num_views, int):
            assert len(views) == self.num_views
        else:
            assert len(views) in self.num_views

        for v, view in enumerate(views):
            # Store the index and other metadata
            view["idx"] = (idx, ar_idx, v)
            view["is_metric_scale"] = self.is_metric_scale
            view["is_synthetic"] = self.is_synthetic

            # Check the depth, intrinsics, and pose data (also other data if present)
            assert "camera_intrinsics" in view
            assert "camera_pose" in view
            assert np.isfinite(view["camera_pose"]).all(), (
                f"NaN or infinite values in camera pose for view {view_name(view)}"
            )
            assert np.isfinite(view["depthmap"]).all(), (
                f"NaN or infinite values in depthmap for view {view_name(view)}"
            )
            assert "valid_mask" not in view
            assert "pts3d" not in view, (
                f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            )
            if "prior_depth_z" in view:
                assert np.isfinite(view["prior_depth_z"]).all(), (
                    f"NaN or infinite values in prior_depth_z for view {view_name(view)}"
                )
            if "non_ambiguous_mask" in view:
                assert np.isfinite(view["non_ambiguous_mask"]).all(), (
                    f"NaN or infinite values in non_ambiguous_mask for view {view_name(view)}"
                )

            # Encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])
            view["data_norm_type"] = self.data_norm_type

            # Compute the pointmaps, raymap and depth along ray
            (
                pts3d,
                valid_mask,
                ray_origins_world,
                ray_directions_world,
                depth_along_ray,
                ray_directions_cam,
                pts3d_cam,
            ) = get_absolute_pointmaps_and_rays_info(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            view["depth_along_ray"] = depth_along_ray
            view["ray_directions_cam"] = ray_directions_cam
            view["pts3d_cam"] = pts3d_cam

            # Compute the prior depth along ray if present
            if "prior_depth_z" in view:
                prior_pts3d, _ = depthmap_to_camera_coordinates(
                    view["prior_depth_z"], view["camera_intrinsics"]
                )
                view["prior_depth_along_ray"] = np.linalg.norm(prior_pts3d, axis=-1)
                view["prior_depth_along_ray"] = view["prior_depth_along_ray"][..., None]
                del view["prior_depth_z"]

            # Convert ambiguous mask dtype to match valid mask dtype
            if "non_ambiguous_mask" in view:
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )
            else:
                ambiguous_mask = view["depthmap"] < 0
                view["non_ambiguous_mask"] = ~ambiguous_mask
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )

            # Check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            # Check shapes
            assert view["depthmap"].shape == view["img"].shape[1:]
            assert view["depthmap"].shape == view["pts3d"].shape[:2]
            assert view["depthmap"].shape == view["valid_mask"].shape
            assert view["depthmap"].shape == view["depth_along_ray"].shape[:2]
            assert view["depthmap"].shape == view["ray_directions_cam"].shape[:2]
            assert view["depthmap"].shape == view["pts3d_cam"].shape[:2]
            if "prior_depth_along_ray" in view:
                assert view["depthmap"].shape == view["prior_depth_along_ray"].shape[:2]
            if "non_ambiguous_mask" in view:
                assert view["depthmap"].shape == view["non_ambiguous_mask"].shape

            # Expand the last dimension of the depthmap
            view["depthmap"] = view["depthmap"][..., None]

            # Append RNG state to the views, this allows to check whether the RNG is in the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            # Compute and store the quaternions and translation for the camera poses
            # Notation is (x, y, z, w) for quaternions
            # This also ensures that the camera poses have a positive determinant (right-handed coordinate system)
            view["camera_pose_quats"] = (
                Rotation.from_matrix(view["camera_pose"][:3, :3])
                .as_quat()
                .astype(view["camera_pose"].dtype)
            )
            view["camera_pose_trans"] = view["camera_pose"][:3, 3].astype(
                view["camera_pose"].dtype
            )

            # Check the pointmaps, rays, depth along ray, and camera pose quaternions and translation to ensure they are finite
            assert np.isfinite(view["pts3d"]).all(), (
                f"NaN in pts3d for view {view_name(view)}"
            )
            assert np.isfinite(view["valid_mask"]).all(), (
                f"NaN in valid_mask for view {view_name(view)}"
            )
            assert np.isfinite(view["depth_along_ray"]).all(), (
                f"NaN in depth_along_ray for view {view_name(view)}"
            )
            assert np.isfinite(view["ray_directions_cam"]).all(), (
                f"NaN in ray_directions_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["pts3d_cam"]).all(), (
                f"NaN in pts3d_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_quats"]).all(), (
                f"NaN in camera_pose_quats for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_trans"]).all(), (
                f"NaN in camera_pose_trans for view {view_name(view)}"
            )
            if "prior_depth_along_ray" in view:
                assert np.isfinite(view["prior_depth_along_ray"]).all(), (
                    f"NaN in prior_depth_along_ray for view {view_name(view)}"
                )

        return views

    def __getitem__(self, idx):
        if self.max_num_retries == 0:
            return self._getitem_fn(idx)

        num_retries = 0
        while num_retries <= self.max_num_retries:
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Error in {type(self).__name__}.__getitem__ for scene_idx={scene_idx}: {e}"
                )

                if num_retries >= self.max_num_retries:
                    print(
                        f"Max retries ({self.max_num_retries}) reached, raising the exception"
                    )
                    raise e

                # Retry with a different scene index
                num_retries += 1
                if isinstance(idx, tuple):
                    # The scene index is the first element of the tuple
                    idx_list = list(idx)
                    idx_list[0] = np.random.randint(0, len(self))
                    idx = tuple(idx_list)
                else:
                    # The scene index is idx
                    idx = np.random.randint(0, len(self))
                scene_idx = idx[0] if isinstance(idx, tuple) else idx
                print(
                    f"Retrying with scene_idx={scene_idx} ({num_retries} of {self.max_num_retries})"
                )


def is_good_type(v):
    """
    Check if a value has an acceptable data type for processing in the dataset.

    Args:
        v: The value to check.

    Returns:
        tuple: A tuple containing:
            - bool: True if the type is acceptable, False otherwise.
            - str or None: Error message if the type is not acceptable, None otherwise.
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    """
    Generate a string identifier for a view based on its dataset, label, and instance.

    Args:
        view (dict): Dictionary containing view information with 'dataset', 'label', and 'instance' keys.
        batch_index (int, optional): Index to select from batched data. Defaults to None.

    Returns:
        str: A formatted string in the form "dataset/label/instance".
    """

    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"
