# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from morphcloud.utils.wai.io import (
    _load_readable,
    _load_scene_meta,
    get_processing_state,
)

logger = logging.getLogger(__name__)


def get_scene_frame_names(
    cfg: dict | object,
    root: Path | str | None = None,
    scene_frames_fn: str | None = None,
    keyframes: bool = True,
) -> dict[str, list[str | float]] | None:
    """
    Retrieve scene frame names based on configuration and optional parameters.

    This function determines the scene frame names by resolving the scene frame file
    and applying any necessary filters based on the provided configuration.

    Args:
        cfg: Configuration object containing settings and parameters.
        root: Optional root directory path. If not provided, it will be fetched from cfg.
        scene_frames_fn: Optional scene frames file name. If not provided, it will be fetched from cfg.
        keyframes: Optional, used only for a video. If True (default), return only keyframes (with camera poses).

    Returns:
        A dictionary mapping scene names to their respective frame names.
    """
    scene_frames_fn = (
        cfg.get("scene_frames_fn") if scene_frames_fn is None else scene_frames_fn
    )
    scene_frame_names = None
    if scene_frames_fn is not None:
        # load scene_frames based on scene_frame file
        scene_frame_names = _resolve_scene_frames_fn(scene_frames_fn)

    scene_names = get_scene_names(
        cfg,
        root=root,
        scene_names=(
            list(scene_frame_names.keys()) if scene_frame_names is not None else None
        ),
    )
    scene_frame_names = _resolve_scene_frame_names(
        cfg,
        scene_names,
        root=root,
        scene_frame_names=scene_frame_names,
        keyframes=keyframes,
    )
    return scene_frame_names


def get_scene_names(
    cfg: dict | object,
    root: Path | str | None = None,
    scene_names: list[str] | None = None,
    shuffle: bool = False,
) -> list[str]:
    """
    Retrieve scene names based on the provided configuration and optional parameters.

    This function determines the scene names by checking the root directory for subdirectories
    and applying any necessary filters based on the provided configuration.

    Args:
        cfg: Configuration object containing settings and parameters.
        root: Optional root directory path. If not provided, it will be fetched from cfg.
        scene_names: Optional list of scene names. If not provided, it will be determined from the root directory.
        shuffle: Optional bool. Default to False. If True, it will return the list of scene names in random order.

    Returns:
        A list of scene names after applying any filters specified in the configuration.
    """
    root = cfg.get("root") if root is None else root
    if root is not None:
        # Check if the root exists
        if not Path(root).exists():
            raise IOError(f"Root directory does not exist: {root}")

        # Check if the root is a directory
        if not Path(root).is_dir():
            raise IOError(f"Root directory is not a directory: {root}")

    if scene_names is None:
        scene_filters = cfg.get("scene_filters")
        if (
            scene_filters
            and len(scene_filters) == 1
            and isinstance(scene_filters[0], list)
            and all(isinstance(entry, str) for entry in scene_filters[0])
        ):
            # Shortcut the scene_names if the scene_filters is only a list of scene names
            scene_names = scene_filters[0]
        else:
            # List all subdirectories in the root as scenes
            scene_names = sorted(
                [entry.name for entry in os.scandir(root) if entry.is_dir()]
            )
    # Filter scenes based on scene_filters
    scene_names = _filter_scenes(root, scene_names, cfg.get("scene_filters"))

    # shuffle the list if needed (in place)
    if shuffle:
        random.shuffle(scene_names)

    return scene_names


def _filter_scenes(
    root: Path | str,
    scene_names: list[str],
    scene_filters: tuple | list | None,
) -> list[str]:
    if scene_filters is None:
        return scene_names

    if not isinstance(scene_filters, (tuple, list)):
        raise ValueError("scene_filters must be a list or tuple")

    for scene_filter in scene_filters:
        if scene_filter in [None, "all"]:
            pass

        elif isinstance(scene_filter, (tuple, list)):
            if len(scene_filter) == 0:
                raise ValueError("scene_filter cannot be empty")

            elif all(isinstance(x, int) for x in scene_filter):
                if len(scene_filter) == 2:
                    # start/end index
                    scene_names = scene_names[scene_filter[0] : scene_filter[1]]
                elif len(scene_filter) == 3:
                    # start/end/step
                    scene_names = scene_names[
                        scene_filter[0] : scene_filter[1] : scene_filter[2]
                    ]
                else:
                    # omegaconf conversion issue (converts strings to integers whenever possible)
                    if str(scene_filter[0]) in scene_names:
                        scene_names = [str(s) for s in scene_filter]
                    else:
                        raise ValueError(
                            "scene_filter format [start_idx, end_idx] or [start_idx, end_idx, step_size] or [scene_name1, scene_name2, ...]"
                        )

            elif all(isinstance(x, str) for x in scene_filter):
                # explicit scene names
                if set(scene_filter).issubset(set(scene_names)):
                    scene_names = list(scene_filter)
                else:
                    logger.warning(
                        f"Scene(s) not available: {set(scene_filter) - set(scene_names)}"
                    )
                    scene_names = list(set(scene_names) & set(scene_filter))
            else:
                raise TypeError(
                    f"Scene filter type not supported: {type(scene_filter)}"
                )

        elif isinstance(scene_filter, dict):
            # reserved key words
            if modality := scene_filter.get("exists"):
                scene_names = [
                    scene_name
                    for scene_name in scene_names
                    if Path(root, scene_name, modality).exists()
                ]

            elif modality := scene_filter.get("exists_not"):
                scene_names = [
                    scene_name
                    for scene_name in scene_names
                    if not Path(root, scene_name, modality).exists()
                ]

            elif process_filter := scene_filter.get("process_state"):
                # filter for where <process_key> has <process_state>
                (process_key, process_state) = process_filter
                filtered_scene_names = []
                for scene_name in scene_names:
                    # load processing state and check for
                    processing_state = get_processing_state(Path(root, scene_name))
                    if "*" in process_key:  # regex matching
                        for process_name in processing_state:
                            if re.match(process_key, process_name):
                                process_key = process_name
                                break
                    if process_key not in processing_state:
                        continue
                    if processing_state[process_key]["state"] == process_state:
                        filtered_scene_names.append(scene_name)
                scene_names = filtered_scene_names

            elif process_filter := scene_filter.get("process_state_not"):
                # filter for where <process_key> does not have <process_state>
                (process_key, process_state) = process_filter
                filtered_scene_names = []
                for scene_name in scene_names:
                    # load processing state and check for
                    try:
                        processing_state = get_processing_state(Path(root, scene_name))
                    except Exception:
                        filtered_scene_names.append(scene_name)
                        continue
                    if "*" in process_key:  # regex matching
                        for process_name in processing_state:
                            if re.match(process_key, process_name):
                                process_key = process_name
                                break
                    if (process_key not in processing_state) or (
                        processing_state[process_key]["state"] != process_state
                    ):
                        filtered_scene_names.append(scene_name)
                scene_names = filtered_scene_names

            else:
                raise ValueError(f"Scene filter not supported: {scene_filter}")

        elif isinstance(scene_filter, str):
            # regex
            scene_names = [
                scene_name
                for scene_name in scene_names
                if re.fullmatch(scene_filter, scene_name)
            ]
        else:
            raise ValueError(f"Scene filter not supported: {scene_filter}")

    return scene_names


def _resolve_scene_frames_fn(scene_frames_fn: str) -> dict[str, list[str] | None]:
    # support for file list in forms of lists or dicts
    # containing scene_names [-> frames]
    scene_frames_list = _load_readable(scene_frames_fn)
    scene_frame_names = {}

    # TODO: The following code seems unreachable as scene_frames_list is always a dict
    if isinstance(scene_frames_list, (list, tuple)):
        for entry in scene_frames_list:
            if isinstance(entry, (tuple, list)):
                if (
                    (len(entry) != 2)
                    or (not isinstance(entry[0], str))
                    or (not isinstance(entry[1], list))
                ):
                    raise NotImplementedError(
                        "Only supports lists of [<scene_name>, [frame_names]]"
                    )
                scene_frame_names[entry[0]] = entry[1]
            elif isinstance(entry, str):
                scene_frame_names[entry] = None
            elif isinstance(entry, dict):
                # scene_name -> frames
                raise NotImplementedError("Dict entry not supported yet")
            else:
                raise IOError(f"File list contains an entry of wrong format: {entry}")

    elif isinstance(scene_frames_list, dict):
        # scene_name -> frames
        for scene_name, frame in scene_frames_list.items():
            if isinstance(frame, (tuple, list)):
                scene_frame_names[scene_name] = frame
            elif isinstance(frame, dict):
                if "frame_names" in frame:
                    scene_frame_names[scene_name] = frame["frame_names"]
                else:
                    raise IOError(f"Scene frames format not supported: {frame}")
            elif frame is None:
                scene_frame_names[scene_name] = frame
            else:
                raise IOError(f"Scene frames format not supported: {frame}")

    else:
        raise IOError(f"Scene frames format not supported: {scene_frames_list}")

    return scene_frame_names


def _resolve_scene_frame_names(
    cfg: dict | object,
    scene_names: list[str],
    root: Path | str | None = None,
    scene_frame_names: dict[str, list[str | float] | None] | None = None,
    keyframes: bool = True,
) -> dict[str, list[str]]:
    root = cfg.get("root") if root is None else root
    if scene_frame_names is not None:
        # restrict to the additional scene-level prefiltering
        scene_frame_names = {
            scene_name: scene_frame_names[scene_name] for scene_name in scene_names
        }
        # dict already loaded, apply additional filters
        for scene_name, frame_names in scene_frame_names.items():
            if frame_names is None:
                scene_meta = _load_scene_meta(
                    Path(
                        root, scene_name, cfg.get("scene_meta_path", "scene_meta.json")
                    )
                )
                frame_names = [frame["frame_name"] for frame in scene_meta["frames"]]
                # TODO: add some logic for video keyframes

            scene_frame_names[scene_name] = _filter_frame_names(
                root, frame_names, scene_name, cfg.get("frame_filters")
            )
    else:
        scene_frame_names = {}
        for scene_name in scene_names:
            scene_meta = _load_scene_meta(
                Path(root, scene_name, cfg.get("scene_meta_path", "scene_meta.json"))
            )
            if not keyframes:
                frame_names = get_video_frames(scene_meta)
                if frame_names is None:
                    keyframes = True
            if keyframes:
                frame_names = [frame["frame_name"] for frame in scene_meta["frames"]]
            frame_names = _filter_frame_names(
                root, frame_names, scene_name, cfg.get("frame_filters")
            )
            scene_frame_names[scene_name] = frame_names
    return scene_frame_names


def _filter_frame_names(
    root: Path | str,
    frame_names: list[str],
    scene_name: str,
    frame_filters: list | tuple | None,
) -> list[str]:
    if frame_filters is None:
        return frame_names

    if not isinstance(frame_filters, (tuple, list)):
        raise ValueError("frame_filters must be a list or tuple")

    for frame_filter in frame_filters:
        if frame_filter in [None, "all"]:
            pass

        elif isinstance(frame_filter, (tuple, list)):
            if len(frame_filter) == 0:
                raise ValueError("frame_filter cannot be empty")

            if isinstance(frame_filter[0], int):
                if len(frame_filter) == 2:
                    # start/end index
                    frame_names = frame_names[frame_filter[0] : frame_filter[1]]

                elif len(frame_filter) == 3:
                    # start/end/step
                    frame_names = frame_names[
                        frame_filter[0] : frame_filter[1] : frame_filter[2]
                    ]

                else:
                    raise ValueError(
                        "frame_filter format [start_idx, end_idx] or [start_idx, end_idx,step_size]"
                    )
            else:
                raise TypeError(
                    f"frame_filter[0] type not supported: {type(frame_filter[0])}"
                )

        elif isinstance(frame_filter, str):
            # reserved key words
            if match := re.match("exists: (.+)", frame_filter):
                modality = match.group(1)
                frame_names = [
                    frame_name
                    for frame_name in frame_names
                    if any(Path(root, scene_name, modality).glob(f"{frame_name}.*"))
                ]

            elif match := re.match("!exists: (.+)", frame_filter):
                modality = match.group(1)
                frame_names = [
                    frame_name
                    for frame_name in frame_names
                    if not any(Path(root, scene_name, modality).glob(f"{frame_name}.*"))
                ]

            else:  # general regex
                frame_names = [
                    frame_name
                    for frame_name in frame_names
                    if re.match(frame_filter, frame_name)
                ]

        else:
            raise ValueError(f"frame_filter type not supported: {type(frame_filter)}")

    return frame_names


def get_video_frames(scene_meta: dict[str, Any]):
    """
    Return names of video frames.
    Args:
        scene_meta: dictionary with scene_meat data.

    Returns:
        A list of video frame names.
    """
    image_modality = [mod for mod in scene_meta["frame_modalities"] if "image" in mod]
    if len(image_modality) > 0:
        image_modality = scene_meta["frame_modalities"][image_modality[0]]
        if "chunks" in image_modality:
            file_list = image_modality["chunks"]
        else:
            file_list = [image_modality]
        frame_names = []
        for chunk in file_list:
            start, end, fps = chunk["start"], chunk["end"], chunk["fps"]
            chunk_frame_names = np.arange(start, end, 1.0 / fps).tolist()
            frame_names += chunk_frame_names
        return frame_names
    return None
