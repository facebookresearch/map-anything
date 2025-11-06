# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from argconf import argconf_parse
from moge.model.v2 import MoGeModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from data_processing.wai_processing.utils.state import (
    SceneProcessLock,
    set_processing_state,
)
from morphcloud.utils.wai.basic_dataset import BasicSceneframeDataset
from morphcloud.utils.wai.camera import intrinsics_to_fov
from morphcloud.utils.wai.core import (
    get_frame,
    load_data,
    nest_modality,
    set_frame,
    store_data,
)
from morphcloud.utils.wai.ops import resize
from morphcloud.utils.wai.scene_frame import get_scene_names

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
logger = logging.getLogger(__name__)


def load_model(model_path: Path, device: str) -> nn.Module:
    """Load specific MoGe model from a given path."""
    return MoGeModel.from_pretrained(str(model_path)).to(device)


def run_moge_on_scene(
    cfg: dict, scene_name: str, model: nn.Module, overwrite: bool = False
) -> None:
    """
    Run MoGe on a given scene.
    Args:
        cfg (dict): Configuration dictionary.
        scene_name (str): Name of the scene to process.
        model: specifi MoGe model in eval mode.
        overwrite (bool): Whether to overwrite existing output.
    Returns:
        None
    """
    cfg.scene_filters = [scene_name]
    scene_root = Path(cfg.root) / scene_name
    scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
    single_scene_dataset = BasicSceneframeDataset(cfg)
    dataloader = DataLoader(
        single_scene_dataset,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=single_scene_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
    )

    # Delete previous generation
    out_path = scene_root / cfg.out_path
    if out_path.exists() and overwrite:
        logger.warning(f"Deleting previous process at : {out_path}")
        shutil.rmtree(out_path)

    normal_opt = False

    for batch in tqdm(dataloader, f"Predicting MoGe ({scene_name})"):
        batch_size = len(batch["frame_name"])
        images = batch["image"]
        intrinsics = batch["intrinsics"]

        _, _, H, W = images.shape
        if cfg.get("resize") is not None:
            if not isinstance(cfg.resize, int) or max(H, W) > cfg.resize:
                # resize only if requested long-size is smaller than the image size or fixed size is specified
                images = resize(images, size=cfg.resize)
                new_H, new_W = images.shape[-2:]
                intrinsics[:, :1] *= new_W / W
                intrinsics[:, 1:2] *= new_H / H
                H, W = new_H, new_W

        (fx, fy) = (
            intrinsics[:, 0, 0],
            intrinsics[:, 1, 1],
        )
        fov_x, _ = intrinsics_to_fov(fx, fy, H, W)
        fov_x = torch.rad2deg(fov_x)
        output = model.infer(
            images,
            fov_x=fov_x,
            resolution_level=cfg.resolution_level,
            use_fp16=False,
        )
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()

        normal_opt |= "normal" in output
        normal = output["normal"].cpu().numpy() if normal_opt else None

        for i in range(batch_size):
            frame_name = batch["frame_name"][i]
            rel_depth_path = f"depth/{cfg.model_name}/{frame_name}.exr"
            rel_mask_path = f"mask/{cfg.model_name}/{frame_name}.png"
            rel_normal_path = f"normals/{cfg.model_name}/{frame_name}.exr"
            store_data(out_path / rel_depth_path, depth[i], "depth")
            store_data(out_path / rel_mask_path, mask[i], "binary")
            if normal_opt:
                store_data(out_path / rel_normal_path, normal[i], "normals")
            # update frame scene_meta
            frame = get_frame(scene_meta, frame_name)
            frame[f"{cfg.model_name}_depth"] = f"{cfg.out_path}/{rel_depth_path}"
            frame[f"{cfg.model_name}_mask"] = f"{cfg.out_path}/{rel_mask_path}"
            if normal_opt:
                frame[f"{cfg.model_name}_normals"] = f"{cfg.out_path}/{rel_normal_path}"
            set_frame(scene_meta, frame_name, frame, sort=True)

    # update frame modalities
    frame_modalities = scene_meta["frame_modalities"]

    # depth
    frame_modalities_depth = nest_modality(frame_modalities, "pred_depth")
    frame_modalities_depth[cfg.model_name] = {
        "frame_key": f"{cfg.model_name}_depth",
        "format": "depth",
    }
    frame_modalities["pred_depth"] = frame_modalities_depth
    # mask
    frame_modalities_mask = nest_modality(frame_modalities, "pred_mask")
    frame_modalities_mask[cfg.model_name] = {
        "frame_key": f"{cfg.model_name}_mask",
        "format": "binary",
    }
    frame_modalities["pred_mask"] = frame_modalities_mask

    # normals
    if normal_opt:
        frame_modalities_normals = nest_modality(frame_modalities, "pred_normals")
        frame_modalities_normals[cfg.model_name] = {
            "frame_key": f"{cfg.model_name}_normals",
            "format": "normals",
        }
        frame_modalities["pred_normals"] = frame_modalities_normals

    # Store new scene_meta
    scene_meta["frame_modalities"] = frame_modalities
    store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    import sys

    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  [{i}]: {arg}")

    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "moge/default.yaml")
    if cfg.get("root") is None:
        raise ValueError(
            "Specify the root via: 'python scripts/run_moge.py root=<root_path>'"
        )

    logger.info("Running MoGe using config:")
    for key, value in dict(cfg).items():
        logger.info(f"  {key}: {value}")

    scene_names = get_scene_names(
        cfg, shuffle=cfg.get("random_scene_processing_order", True)
    )

    model_path = Path(cfg.model_path)
    model = load_model(model_path, cfg.device).eval()

    logger.info(f"Processing: {len(scene_names)} scenes")
    logger.debug(f"scene_names = {scene_names}")

    for scene_name in tqdm(scene_names, "Processing scenes"):
        try:
            scene_root = Path(cfg.root) / scene_name
            with SceneProcessLock(scene_root):
                logger.info(f"Processing: {scene_name}")
                set_processing_state(scene_root, "moge", "running")
                run_moge_on_scene(cfg, scene_name, model, cfg.overwrite)
                set_processing_state(scene_root, "moge", "finished")
        except Exception:
            logger.error(f"Running MoGe failed on scene '{scene_name}'")
            trace_message = traceback.format_exc()
            logger.error(trace_message)
            set_processing_state(scene_root, "moge", "failed", message=trace_message)
            continue
