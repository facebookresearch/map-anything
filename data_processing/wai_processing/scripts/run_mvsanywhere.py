# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import datetime
import logging
import random
import shutil
import string
import traceback
from multiprocessing import Manager
from pathlib import Path

import torch
from argconf import argconf_parse
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from data_processing.wai_processing.utils.state import (
    SceneProcessLock,
    set_processing_state,
)
from morphcloud.utils.wai.core import (
    get_frame,
    load_data,
    nest_modality,
    set_frame,
    store_data,
)
from morphcloud.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger("mvsanywhere")


def load_model(mvsanywhere_path: str, ckpt_path: str, opts, device: str = "cuda"):
    """
    Loads a model for inference from the specified mvsanywhere repository path.

    Args:
        mvsanywhere_path (str): Path to the mvsanywhere repository.
        ckpt_path (str): Path to the model checkpoint file (currently unused).
        opts: Options or configurations for loading the model.
        device (str, optional): Device to load the model on, defaults to "cuda".

    Returns:
        The loaded model set to evaluation mode on the specified device.
    """
    if not Path(mvsanywhere_path).exists():
        raise RuntimeError(f"mvsanywhere repo not found at: {mvsanywhere_path}")

    sys.path.append(str(mvsanywhere_path))
    from mvsanywhere.utils.model_utils import get_model_class, load_model_inference

    model_class_to_use = get_model_class(opts)
    model = load_model_inference(opts, model_class_to_use)
    model = model.to(device).eval()

    return model


def to_gpu(input_dict: dict, key_ignores: list = None) -> dict:
    """Moves tensors in the input dict to the GPU.

    Args:
        input_dict: Dictionary containing tensors to move to GPU
        key_ignores: List of keys to ignore when moving to GPU

    Returns:
        The input dictionary with tensors moved to GPU
    """
    if key_ignores is None:
        key_ignores = []

    for k, v in input_dict.items():
        if k not in key_ignores:
            input_dict[k] = v.cuda().float()
    return input_dict


def prepare_dataset_options(cfg, opts):
    """Prepares and configures dataset options for processing.

    This function sets up the necessary paths and configurations for
    the mvsanywhere dataset processing, including creating temporary directories and
    files, and modifying options based on the provided configuration.

    Args:
        cfg: Configuration object containing model and output paths.
        opts: Options object containing dataset and processing options.

    Returns:
        Updated dataset options with paths and configurations set.
    """

    sys.path.append(str(cfg.model_path))
    from mvsanywhere.tools.tuple_generator import crawl_subprocess_long

    dataset_opts = opts.datasets[0]

    # get the parent path.
    parent_path = Path(opts.scan_parent_directory)
    scan_name = opts.scan_name

    # create a tmp directory for dataset files.
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    tmp_metadata_folder = Path("/tmp") / f"fmvs_{random_string}_{current_time}"

    # make a scans file and a folder for the tuples
    tmp_metadata_folder.mkdir(parents=True, exist_ok=True)
    dataset_scan_split_file = tmp_metadata_folder / "scans.txt"
    tuple_info_file_location = tmp_metadata_folder / "tuples"
    tuple_info_file_location.mkdir(parents=True, exist_ok=True)

    # check the tuple type
    frame_tuple_type = (
        "dense_offline"
        if dataset_opts.frame_tuple_type is None
        else dataset_opts.frame_tuple_type
    )

    # make a meaningful name for mv_tuple_file_suffix
    mv_tuple_file_suffix = f"_eight_view_{frame_tuple_type}.txt"

    # write the scan folder name in the scans.txt file.
    with open(dataset_scan_split_file, "w") as f:
        f.write(scan_name + "\n")

    single_debug_scan_id = scan_name

    # modify the opts file to include all the above
    dataset_opts.dataset_path = str(parent_path)
    dataset_opts.single_debug_scan_id = single_debug_scan_id
    dataset_opts.tuple_info_file_location = str(tuple_info_file_location)
    dataset_opts.dataset_scan_split_file = str(dataset_scan_split_file)
    dataset_opts.mv_tuple_file_suffix = mv_tuple_file_suffix

    opts.dataset_opts = dataset_opts

    # compute tuples
    tuples = crawl_subprocess_long(
        opts,
        single_debug_scan_id,
        0,
        Manager().Value("i", 0),
    )

    # save tuples
    with open(
        tuple_info_file_location / f"{dataset_opts.split}{mv_tuple_file_suffix}", "w"
    ) as f:
        for line in tuples:
            f.write(line + "\n")

    return dataset_opts


def dataset_preparation(cfg, scene_name: str, overwrite, scene_root, opts):
    # Delete previous generation
    out_path = scene_root / cfg.out_path
    logger.info(f"writing to out_path: {out_path}")
    if out_path.exists():
        if overwrite:
            logger.warning(f"deleting existing: {out_path}")
            shutil.rmtree(out_path)
        else:
            logger.warning(
                f"out_path: {out_path} already exists, set overwrite to True to overwrite"
            )

    scene_meta = load_data(Path(scene_root, "scene_meta.json"), "scene_meta")

    sys.path.append(str(cfg.model_path))

    from mvsanywhere.utils.dataset_utils import get_dataset

    # get dataset
    dataset_opts = prepare_dataset_options(cfg, opts)
    opts.datasets[0] = dataset_opts

    assert len(opts.datasets) == 1, (
        f"Expected only one dataset but got {len(opts.datasets)}"
    )
    dataset_opts = opts.datasets[0]

    # TODO: mvsanywhere specific options, can probably be fixed in the config file and function call avoided
    dataset_class, scans = get_dataset(
        dataset_opts.dataset,
        dataset_opts.dataset_scan_split_file,
        opts.single_debug_scan_id,
    )
    scan = scans[0]  # only one "scan" is used, so we can just take the first one
    return (dataset_class, dataset_opts, scan, out_path, scene_meta)


def run_mvsanywhere_on_scene(cfg, scene_name: str, opts, overwrite=False):
    # Create a dataloader that only parses a single scene.
    # This ensures that every loaded frame belongs to this scene.
    cfg.scene_filters = [scene_name]
    scene_root = Path(cfg.root) / scene_name
    (dataset_class, dataset_opts, scan, out_path, scene_meta) = dataset_preparation(
        cfg, scene_name, overwrite, scene_root, opts
    )

    with torch.inference_mode():
        mvsa_dataset = dataset_class(
            dataset_opts.dataset_path,
            split=dataset_opts.split,
            mv_tuple_file_suffix=dataset_opts.mv_tuple_file_suffix,
            limit_to_scan_id=scan,
            include_full_res_depth=True,
            tuple_info_file_location=dataset_opts.tuple_info_file_location,
            num_images_in_tuple=None,
            shuffle_tuple=opts.shuffle_tuple,
            include_high_res_color=True,
            include_full_depth_K=True,
            skip_frames=opts.skip_frames,
            skip_to_frame=opts.skip_to_frame,
            image_width=opts.image_width,
            image_height=opts.image_height,
            pass_frame_id=True,
            disable_flip=True,
            rotate_images=opts.rotate_images,
            matching_scale=opts.matching_scale,
            prediction_scale=opts.prediction_scale,
            prediction_num_scales=opts.prediction_num_scales,
        )

        dataloader = DataLoader(
            mvsa_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

        # Iterate over all images and perform depth estimation.
        for batch in tqdm(dataloader, f"Predicting MVSAnywhere ({scene_name})"):
            cur_data, src_data = batch
            batch_size = len(cur_data["frame_id_string"])
            cur_data = to_gpu(cur_data, key_ignores=["frame_id_string", "frame_name"])
            src_data = to_gpu(src_data, key_ignores=["frame_id_string", "frame_name"])

            outputs = model(
                phase="test",
                cur_data=cur_data,
                src_data=src_data,
                unbatched_matching_encoder_forward=(not opts.fast_cost_volume),
                return_mask=True,
                num_refinement_steps=1,
            )
            # Store outputs
            for b in range(batch_size):
                frame_name = cur_data["frame_name"][b]
                rel_depth_path = f"depth/{frame_name}.exr"
                store_data(
                    out_path / rel_depth_path,
                    outputs["depth_pred_s0_b1hw"][b].squeeze(),
                    "depth",
                )

                frame = get_frame(scene_meta, frame_name)
                frame[f"{cfg.model_name}_depth"] = f"{cfg.out_path}/{rel_depth_path}"
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
        scene_meta["frame_modalities"] = frame_modalities

        # Store new scene_meta
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")


def get_mvsa_options(cfg):
    sys.path.append(str(cfg.model_path))
    import mvsanywhere.options as mvsa_options

    option_handler = mvsa_options.OptionsHandler()
    opts = option_handler.options

    opts.name = "mvsanywhere"
    opts.model_type = "depth_model"
    opts.output_base_path = (
        cfg.out_path
    )  # Assuming BASE_OUTPUT_DIR is defined elsewhere
    opts.config_file = cfg.config_file
    opts.load_weights_from_checkpoint = cfg.ckpt_path
    opts.data_config_file = cfg.data_config_file
    opts.scan_parent_directory = cfg.root
    opts.fast_cost_volume = True  # Flag without value is typically set to True

    # add from standard config first
    if opts.config_file is not None:
        config_options = option_handler.load_options_from_yaml(opts.config_file)
        option_handler.merge_config_options(config_options)

    # then merge from a data config
    if opts.data_config_file is not None:
        opts.datasets = [
            option_handler.load_options_from_yaml(data_config)
            for data_config in opts.data_config_file.split(":")
        ]
    else:
        # no config has been supplied. Let's hope that we have required
        # arguments through command line.
        logger.info("Not reading from a config_file.")
        config_options = None

    # make sure this is set at the end, so the final settings are taken from the wai config file:
    opts.val_image_height = cfg.val_image_height
    opts.val_image_width = cfg.val_image_width
    opts.image_height = cfg.image_height
    opts.image_width = cfg.image_width
    opts.prediction_scale = cfg.prediction_scale
    opts.num_workers = cfg.num_workers
    opts.batch_size = cfg.batch_size
    logger.info("MVSAnywhere options")
    option_handler.pretty_print_options()

    return opts


if __name__ == "__main__":
    import sys

    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  [{i}]: {arg}")

    # TODO: each dataset needs a different config (adapt aspect ratio of predicted depth), discuss solutions to have one config for all
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "mvs_anywhere/default.yaml")
    if cfg.get("root") is None:
        raise ValueError(
            "Specify the root via: 'python scripts/run_mvsanywhere.py root=<root_path>'"
        )
    else:
        logger.info(f"root path is: {cfg.get('root')}")
    logger.info("Running MVSAnywhere using config:")
    for key, value in dict(cfg).items():
        logger.info(f"  {key}: {value}")

    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled!")

    scene_names = get_scene_names(
        cfg, shuffle=cfg.get("random_scene_processing_order", True)
    )

    opts = get_mvsa_options(cfg)

    model = load_model(cfg.model_path, cfg.ckpt_path, opts, device="cuda")
    logger.info(f"Processing: {len(scene_names)} scenes")
    logger.debug(f"scene_names = {scene_names}")
    for scene_name in tqdm(scene_names, "Processing scenes"):
        try:
            scene_root = Path(cfg.root) / scene_name
            with SceneProcessLock(scene_root):
                opts.scan_name = scene_name  # Assuming SCAN_NAME is defined elsewhere
                logger.info(f"Processing: {scene_name}")
                set_processing_state(scene_root, "mvsanywhere", "running")
                run_mvsanywhere_on_scene(cfg, scene_name, opts, overwrite=overwrite)
                set_processing_state(scene_root, "mvsanywhere", "finished")
        except Exception:
            logger.error(f"Running mvsanywhere failed on scene '{scene_name}'")
            trace_message = traceback.format_exc()
            logger.error(trace_message)
            set_processing_state(
                scene_root, "mvsanywhere", "failed", message=trace_message
            )
            continue
