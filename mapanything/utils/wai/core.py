# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
This utils script contains PORTAGE of wai-core core methods for MapAnything.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mapanything.utils.wai.camera import (
    CAMERA_KEYS,
    convert_camera_coeffs_to_pinhole_matrix,
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from mapanything.utils.wai.io import _get_method, _load_scene_meta
from mapanything.utils.wai.ops import crop

logger = logging.getLogger(__name__)

WAI_COLORMAP_PATH = Path(__file__).parent / "colormaps"


def load_data(fname: str | Path, format_type: str | None = None, **kwargs) -> Any:
    """
    Loads data from a file using the appropriate method based on the file format.

    Args:
        fname (str or Path): The filename or path to load data from.
        format_type (str, optional): The format type of the data. If None, it will be inferred from the file extension if possible.
            Supported formats include: 'readable', 'scalar', 'image', 'binary', 'depth', 'normals',
            'numpy', 'ptz', 'mmap', 'scene_meta', 'labeled_image', 'mesh', 'labeled_mesh', 'caption', "latents".
        **kwargs: Additional keyword arguments to pass to the loading method.

    Returns:
        The loaded data in the format returned by the specific loading method.

    Raises:
        ValueError: If the format cannot be inferred from the file extension.
        NotImplementedError: If the specified format is not supported.
        FileExistsError: If the file does not exist.
    """
    load_method = _get_method(fname, format_type, load=True)
    return load_method(fname, **kwargs)


def store_data(
    fname: str | Path,
    data: Any,
    format_type: str | None = None,
    **kwargs,
) -> Any:
    """
    Stores data to a file using the appropriate method based on the file format.

    Args:
        fname (str or Path): The filename or path to store data to.
        data: The data to be stored.
        format_type (str, optional): The format type of the data. If None, it will be inferred from the file extension.
        **kwargs: Additional keyword arguments to pass to the storing method.

    Returns:
        The result of the storing method, which may vary depending on the method used.
    """
    store_method = _get_method(fname, format_type, load=False)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    return store_method(fname, data, **kwargs)


def get_frame(
    scene_meta: dict[str, Any],
    frame_key: int | str | float,
) -> dict[str, Any]:
    """
    Get a frame from scene_meta based on name or index.

    Args:
        scene_meta: Dictionary containing scene metadata
        frame_key: Either a string (frame name) or integer (frame index) or float (video timestamp)

    Returns:
        The frame data (dict)
    """
    frame_idx = get_frame_index(scene_meta, frame_key)
    if isinstance(frame_idx, int):
        frame = scene_meta["frames"][frame_idx]
        frame["_is_interpolated"] = False
    else:
        frame = {}
        frame["frame_name"] = frame_key
        left = int(frame_idx)  # it's floor operation
        assert left >= 0 and left < (len(scene_meta["frames"]) - 1), "Wrong index"
        frame_left = scene_meta["frames"][left]
        frame_right = scene_meta["frames"][left + 1]
        # Interpolate intrinsics and extrinsics
        frame["transform_matrix"] = interpolate_extrinsics(
            frame_left["transform_matrix"],
            frame_right["transform_matrix"],
            frame_idx - left,
        )
        frame.update(
            interpolate_intrinsics(
                frame_left,
                frame_right,
                frame_idx - left,
            )
        )
        frame["_is_interpolated"] = True
    return frame


def get_intrinsics(
    scene_meta,
    frame_key,
    fmt: str = "torch",
) -> torch.Tensor | np.ndarray | list:
    frame = get_frame(scene_meta, frame_key)
    return convert_camera_coeffs_to_pinhole_matrix(scene_meta, frame, fmt=fmt)


def get_extrinsics(
    scene_meta,
    frame_key,
    fmt: str = "torch",
) -> torch.Tensor | np.ndarray | list | None:
    frame = get_frame(scene_meta, frame_key)
    if "transform_matrix" in frame:
        if fmt == "torch":
            return torch.tensor(frame["transform_matrix"]).reshape(4, 4).float()
        elif fmt == "np":
            return np.array(frame["transform_matrix"]).reshape(4, 4)
        return frame["transform_matrix"]
    else:
        # TODO: should not happen if we enable interpolation
        return None


def get_frame_index(
    scene_meta: dict[str, Any],
    frame_key: int | str | float,
    frame_index_threshold_sec: float = 1e-4,
    distance_threshold_sec: float = 2.0,
) -> int | float:
    """
    Returns the frame index from scene_meta based on name (str) or index (int) or sub-frame index (float).

    Args:
        scene_meta: Dictionary containing scene metadata
        frame_key: Either a string (frame name) or integer (frame index) or float (sub-frame index)
        frame_index_threshold_sec: A threshold for nearest neighbor clipping for indexes (in seconds).
                                   Default is 1e-4, which is 10000 fps.
        distance_th: A threshold for maximum distance between interpolated frames (in seconds).

    Returns:
        Frame index (int)

    Raises:
        ValueError: If frame_key is not a string or integer or float
    """
    if isinstance(frame_key, str):
        try:
            return scene_meta["frame_names"][frame_key]
        except KeyError as err:
            error_message = (
                f"Frame name not found: {frame_key} - "
                f"Please verify scene_meta.json of scene: {scene_meta['dataset_name']}/{scene_meta['scene_name']}"
            )
            logger.error(error_message)
            raise KeyError(error_message) from err

    if isinstance(frame_key, int):
        return frame_key

    if isinstance(frame_key, float):
        # If exact hit
        if frame_key in scene_meta["frame_names"]:
            return scene_meta["frame_names"][frame_key]

        frame_names = sorted(list(scene_meta["frame_names"].keys()))
        distances = np.array([frm - frame_key for frm in frame_names])
        left = int(np.nonzero(distances <= 0)[0][-1])
        right = left + 1

        # The last frame or rounding errors
        if (
            left == distances.shape[0] - 1
            or abs(distances[left]) < frame_index_threshold_sec
        ):
            return scene_meta["frame_names"][frame_names[int(left)]]
        if abs(distances[right]) < frame_index_threshold_sec:
            return scene_meta["frame_names"][frame_names[int(right)]]

        interpolation_distance = distances[right] - distances[left]
        if interpolation_distance > distance_threshold_sec:
            raise ValueError(
                f"Frame interpolation is forbidden for distances larger than {distance_threshold_sec}."
            )
        alpha = -distances[left] / interpolation_distance

        return scene_meta["frame_names"][frame_names[int(left)]] + alpha

    raise ValueError(f"Frame key type not supported: {frame_key} ({type(frame_key)}).")


def load_modality_data(
    scene_root: Path | str,
    results: dict[str, Any],
    modality_dict: dict[str, Any],
    modality: str,
    frame: dict[str, Any] | None = None,
    fmt: str = "torch",
) -> dict[str, Any]:
    """
    Processes a modality by loading data from a specified path and updating the results dictionary.
    This function extracts the format and path from the given modality dictionary, loads the data
    from the specified path, and updates the results dictionary with the loaded data.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        results (dict): A dictionary to store the loaded modality data and optional frame path.
        modality_dict (dict): A dictionary containing the modality information, including 'format'
            and the path to the data.
        modality (str): The key under which the loaded modality data will be stored in the results.
        frame (dict, optional): A dictionary containing frame information. If provided, that means we are loading
        frame modalities, otherwise it is scene modalities.

    Returns:
        dict: The updated results dictionary containing the loaded modality data.
    """
    modality_format = modality_dict["format"]

    # The modality is stored as a video
    if "video" in modality_format:
        assert isinstance(frame["frame_name"], float), "frame_name should be float"
        video_file = None
        if "chunks" in modality_dict:
            video_list = modality_dict["chunks"]
            # Get the correct chunk of the video
            for video_chunk in video_list:
                if video_chunk["start"] <= frame["frame_name"] <= video_chunk["end"]:
                    video_file = video_chunk
                    break
        else:
            # There is only one video (no chunks)
            video_file = modality_dict
            if "start" not in video_file:
                video_file["start"] = 0
            if "end" not in video_file:
                video_file["end"] = float("inf")
            if not (video_file["start"] <= frame["frame_name"] <= video_file["end"]):
                video_file = None

        # This timestamp is not available in any of the chunks
        if video_file is None:
            frame_name = frame["frame_name"]
            logger.warning(
                f"Modality {modality} ({modality_format}) is not available at time {frame_name}"
            )
            return results

        # Load the modality from the video
        loaded_modality = load_data(
            Path(scene_root, video_file["file"]),
            modality_format,
            frame_key=frame["frame_name"] - video_file["start"],
        )

        if "bbox" in video_file:
            loaded_modality = crop(loaded_modality, video_file["bbox"])

        if loaded_modality is not None:
            results[modality] = loaded_modality

        if frame:
            results[f"{modality}_fname"] = video_file["file"]
    else:
        modality_path = [v for k, v in modality_dict.items() if k != "format"][0]
        if frame:
            if modality_path in frame:
                fname = frame[modality_path]
            else:
                fname = None
        else:
            fname = modality_path
        if fname is not None:
            loaded_modality = load_data(
                Path(scene_root, fname),
                modality_format,
                frame_key=frame["frame_name"] if frame else None,
                fmt=fmt,
            )
            results[modality] = loaded_modality
            if frame:
                results[f"{modality}_fname"] = frame[modality_path]
    return results


def load_modality(
    scene_root: Path | str,
    modality_meta: dict[str, Any],
    modality: str,
    frame: dict[str, Any] | None = None,
    fmt: str = "torch",
) -> dict[str, Any]:
    """
    Loads modality data based on the provided metadata and updates the results dictionary.
    This function navigates through the modality metadata to find the specified modality,
    then loads the data for each modality found.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        modality_meta (dict): A nested dictionary containing metadata for various modalities.
        modality (str): A string representing the path to the desired modality within the metadata,
            using '/' as a separator for nested keys.
        frame (dict, optional): A dictionary containing frame information. If provided, we are operating
        on frame modalities, otherwise it is scene modalities.

    Returns:
        dict: A dictionary containing the loaded modality data.
    """
    results = {}
    # support for nested modalities like "pred_depth/metric3dv2"
    modality_keys = modality.split("/")
    current_modality = modality_meta
    for key in modality_keys:
        try:
            current_modality = current_modality[key]
        except KeyError as err:
            error_message = (
                f"Modality '{err.args[0]}' not found in modalities metadata. "
                f"Please verify the scene_meta.json and the provided modalities in {scene_root}."
            )
            logger.error(error_message)
            raise KeyError(error_message) from err
    if "format" in current_modality:
        results = load_modality_data(
            scene_root, results, current_modality, modality, frame, fmt=fmt
        )
    else:
        # nested modality, return last by default
        logger.warning("Nested modality, returning last by default")
        key = next(reversed(current_modality.keys()))
        results = load_modality_data(
            scene_root, results, current_modality[key], modality, frame, fmt=fmt
        )
    return results


def load_frame(
    scene_root: Path | str,
    frame_key: int | str | float,
    modalities: str | list[str] | None = None,
    scene_meta: dict[str, Any] | None = None,
    load_intrinsics: bool = True,
    load_extrinsics: bool = True,
    fmt: str = "torch",
    interpolate: bool = False,
) -> dict[str, Any]:
    """
    Load a single frame from a scene with specified modalities.

    Args:
        scene_root (str or Path): The root directory of the scene where the data is located.
        frame_key (int or str or float): Either a string (frame name) or integer (frame index) or float (video timestamp).
        modalities (str or list[str], optional): The modality or list of modalities to load.
            If None, only basic frame information is loaded.
        scene_meta (dict, optional): Dictionary containing scene metadata. If None, it will be loaded
            from scene_meta.json in the scene_root.
        interpolate (bool, optional): Allow interpolating frames?

    Returns:
        dict: A dictionary containing the loaded frame data with the requested modalities.
    """
    scene_root = Path(scene_root)
    if scene_meta is None:
        scene_meta = _load_scene_meta(scene_root / "scene_meta.json")
    frame = get_frame(scene_meta, frame_key)
    # compact, standardized frame representation
    wai_frame = {}
    if load_extrinsics:
        extrinsics = get_extrinsics(
            scene_meta,
            frame_key,
            fmt=fmt,
        )
        if extrinsics is not None:
            wai_frame["extrinsics"] = extrinsics
    if load_intrinsics:
        camera_model = frame.get("camera_model", scene_meta.get("camera_model"))
        wai_frame["camera_model"] = camera_model
        if camera_model == "PINHOLE":
            wai_frame["intrinsics"] = get_intrinsics(scene_meta, frame_key, fmt=fmt)
        elif camera_model in ["OPENCV", "OPENCV_FISHEYE"]:
            # optional per-frame intrinsics
            for camera_key in CAMERA_KEYS:
                if camera_key in frame:
                    wai_frame[camera_key] = float(frame[camera_key])
                elif camera_key in scene_meta:
                    wai_frame[camera_key] = float(scene_meta[camera_key])
        else:
            error_message = (
                f"Camera model not supported: {camera_model} - "
                f"Please verify scene_meta.json of scene: {scene_meta['dataset_name']}/{scene_meta['scene_name']}"
            )
            logger.error(error_message)
            raise NotImplementedError(error_message)
    wai_frame["w"] = frame.get("w", scene_meta["w"] if "w" in scene_meta else None)
    wai_frame["h"] = frame.get("h", scene_meta["h"] if "h" in scene_meta else None)
    wai_frame["frame_name"] = frame["frame_name"]
    wai_frame["frame_idx"] = get_frame_index(scene_meta, frame_key)
    wai_frame["_is_interpolated"] = frame["_is_interpolated"]

    if modalities is not None:
        if isinstance(modalities, str):
            modalities = [modalities]
        for modality in modalities:
            # Handle regex patterns in modality
            if any(char in modality for char in ".|*+?()[]{}^$\\"):
                # This is a regex pattern
                pattern = re.compile(modality)
                matching_modalities = [
                    m for m in scene_meta["frame_modalities"] if pattern.match(m)
                ]
                if not matching_modalities:
                    raise ValueError(
                        f"No modalities match the pattern: {modality} in scene: {scene_root}"
                    )
                # Use the first matching modality
                modality = matching_modalities[0]
            current_modalities = load_modality(
                scene_root, scene_meta["frame_modalities"], modality, frame, fmt=fmt
            )
            wai_frame.update(current_modalities)

    return wai_frame


def set_frame(
    scene_meta: dict[str, Any],
    frame_key: int | str,
    new_frame: dict[str, Any],
    sort: bool = False,
) -> dict[str, Any]:
    """
    Replace a frame in scene_meta with a new frame.

    Args:
        scene_meta: Dictionary containing scene metadata.
        frame_key: Either a string (frame name) or integer (frame index).
        new_frame: New frame data to replace the existing frame.
        sort: If True, sort the keys in the new_frame dictionary.

    Returns:
        Updated scene_meta dictionary.
    """
    frame_idx = get_frame_index(scene_meta, frame_key)
    if isinstance(frame_idx, float):
        raise ValueError(
            f"Setting frame for sub-frame frame_key is not supported: {frame_key} ({type(frame_key)})."
        )
    if sort:
        new_frame = {k: new_frame[k] for k in sorted(new_frame)}
    scene_meta["frames"][frame_idx] = new_frame
    return scene_meta


def nest_modality(
    frame_modalities: dict[str, Any],
    modality_name: str,
) -> dict[str, Any]:
    """
    Converts a flat modality structure into a nested one based on the modality name.

    Args:
        frame_modalities (dict): Dictionary containing frame modalities.
        modality_name (str): The name of the modality to nest.

    Returns:
        dict: A dictionary with the nested modality structure.
    """
    frame_modality = {}
    if modality_name in frame_modalities:
        frame_modality = frame_modalities[modality_name]
        if "frame_key" in frame_modality:
            # required for backwards compatibility
            # converting non-nested format into nested one based on name
            modality_name = frame_modality["frame_key"].split("_")[0]
            frame_modality = {modality_name: frame_modality}
    return frame_modality
