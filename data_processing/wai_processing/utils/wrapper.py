# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import traceback
from pathlib import Path
from shutil import rmtree
from typing import Callable

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from data_processing.wai_processing.utils.state import set_processing_state
from morphcloud.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger(__name__)


def get_original_scene_names(cfg):
    """
    Get all original scene ids that should be converted:
    1. Get all available scenes from the source / original dataset
    2. Filter these scenes with the scene filters set in the config
        yaml, usually filter out scenes with conversion status
        'finished'.
    """
    scene_names = get_scene_names(cfg, root=cfg.original_root)
    scene_names = get_scene_names(cfg, scene_names=scene_names)
    return scene_names


def convert_scenes_wrapper(
    converter_func: Callable,
    cfg: DictConfig,
    get_original_scene_names_func: Callable | None = None,
    # arbitrary **kwargs forwarded to the converter_func, example in dl3dv conversion
    **kwargs,
):
    """
    Wrapper function for converting  that handles folder creation etc.
    Args:
        converter_func (callable): Function to use for the per scene conversion.
        cfg (dict): Configuration dictionary.
        **kwargs: Additional keyword arguments to pass to the converter function.
    """
    if get_original_scene_names_func is None:
        get_original_scene_names_func = get_original_scene_names

    scene_names = get_original_scene_names_func(cfg)
    logger.info(f"Processing: {len(scene_names)} scenes")
    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled")
    succ_scenes = []

    for scene_name in tqdm(sorted(scene_names)):
        scene_outpath = Path(cfg.root) / scene_name
        if scene_outpath.exists():
            if overwrite:
                rmtree(scene_outpath)
            else:
                raise FileExistsError(
                    f"Output path already exists: {scene_outpath} - set overwrite=True if this is safe"
                )
        scene_outpath.mkdir(parents=True, exist_ok=True)
        set_processing_state(scene_outpath, "conversion", "running")

        try:
            return_state = converter_func(cfg, scene_name, **kwargs)
            if (
                (return_state is not None)
                and (return_state != "finished")
                and (return_state[0] != "finished")
            ):
                # TODO: Should we add handling of whether scenes which return a non-standard state count as failed, successful
                # or only the returned state. Right now always assuming the last case.
                if type(return_state) is str:
                    # Return state is just a string of the state
                    set_processing_state(scene_outpath, return_state)
                else:
                    # return state is a tuple of: state, message
                    assert len(return_state) == 2, (
                        "Expected return state to either be a string, "
                        "or a tuple of length two with (<module_state>, <message>)"
                    )
                    set_processing_state(
                        scene_outpath, return_state[0], return_state[1]
                    )
                logger.warning(f"Scene {scene_name} finished with state: {scene_name}")
            else:
                set_processing_state(scene_outpath, "conversion", "finished")
                for processing_key, processing_val in cfg.get(
                    "additional_processing_states", []
                ):
                    set_processing_state(scene_outpath, processing_key, processing_val)
                succ_scenes.append(scene_name)
        except Exception:
            trace_message = traceback.format_exc()
            logger.warning(
                f"\nConversion failed on scene: {scene_name}"
                f"\nError message: \n{trace_message}\n"
            )
            set_processing_state(
                scene_outpath, "conversion", "failed", message=trace_message
            )

    logger.info(f"Finished converting {len(succ_scenes)} / {len(scene_names)} scenes.")
