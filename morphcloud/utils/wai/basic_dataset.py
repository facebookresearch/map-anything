# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any

import torch
from box import Box

from morphcloud.utils.wai.core import get_frame_index, load_data, load_frame
from morphcloud.utils.wai.ops import stack
from morphcloud.utils.wai.scene_frame import get_scene_frame_names


class BasicSceneframeDataset(torch.utils.data.Dataset):
    """Basic wai dataset to iterative over frames of scenes"""

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return stack(batch)

    def __init__(
        self,
        cfg: Box,
    ):
        """
        Initialize the BasicSceneframeDataset.

        Args:
            cfg (Box): Configuration object containing dataset parameters including:
                - root: Root directory containing scene data
                - frame_modalities: List of modalities to load for each frame
                - key_remap: Optional dictionary mapping original keys to new keys
        """
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root
        keyframes = cfg.get("use_keyframes", True)
        self.scene_frame_names = get_scene_frame_names(cfg, keyframes=keyframes)
        self.scene_frame_list = [
            (scene_name, frame_name)
            for scene_name, frame_names in self.scene_frame_names.items()
            for frame_name in frame_names
        ]
        self._scene_cache = {}

    def __len__(self):
        """
        Get the total number of scene-frame pairs in the dataset.

        Returns:
            int: The number of scene-frame pairs.
        """
        return len(self.scene_frame_list)

    def _load_scene(self, scene_name: str) -> dict[str, Any]:
        """
        Load scene data for a given scene name.

        Args:
            scene_name (str): The name of the scene to load.

        Returns:
            dict: A dictionary containing scene data, including scene metadata.
        """
        # load scene data
        scene_data = {}
        scene_data["meta"] = load_data(
            Path(
                self.root,
                scene_name,
                self.cfg.get("scene_meta_path", "scene_meta.json"),
            ),
            "scene_meta",
        )

        return scene_data

    def _load_scene_frame(
        self, scene_name: str, frame_name: str | float
    ) -> dict[str, Any]:
        """
        Load data for a specific frame from a specific scene.

        This method loads scene data if not already cached, then loads the specified frame
        from that scene with the modalities specified in the configuration.

        Args:
            scene_name (str): The name of the scene containing the frame.
            frame_name (str or float): The name/timestamp of the frame to load.

        Returns:
            dict: A dictionary containing the loaded frame data with requested modalities.
        """
        scene_frame_data = {}
        if not (scene_data := self._scene_cache.get(scene_name)):
            scene_data = self._load_scene(scene_name)
            # for now only cache the last scene
            self._scene_cache = {}
            self._scene_cache[scene_name] = scene_data

        frame_idx = get_frame_index(scene_data["meta"], frame_name)

        scene_frame_data["scene_name"] = scene_name
        scene_frame_data["frame_name"] = frame_name
        scene_frame_data["scene_path"] = str(Path(self.root, scene_name))
        scene_frame_data["frame_idx"] = frame_idx
        scene_frame_data.update(
            load_frame(
                Path(self.root, scene_name),
                frame_name,
                modalities=self.cfg.frame_modalities,
                scene_meta=scene_data["meta"],
            )
        )
        # Remap key names
        for key, new_key in self.cfg.get("key_remap", {}).items():
            if key in scene_frame_data:
                scene_frame_data[new_key] = scene_frame_data.pop(key)

        return scene_frame_data

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a specific scene-frame pair by index.

        Args:
            index (int): The index of the scene-frame pair to retrieve.

        Returns:
            dict: A dictionary containing the loaded frame data with requested modalities.
        """
        scene_frame = self._load_scene_frame(*self.scene_frame_list[index])
        return scene_frame
