# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from math import ceil
from pathlib import Path

import spod
from argconf import argconf_parse
from wai_processing.launch.launch_utils import (
    _escape_scene_names,
    import_function_from_path,
    parse_string_to_dict,
)
from wai_processing.utils.globals import (
    WAI_PROC_CONFIG_PATH,
    WAI_PROC_MAIN_PATH,
    WAI_PROC_SCRIPT_PATH,
    WAI_SPOD_RUNS_PATH,
)
from wai_processing.utils.state import get_commit_hash

from morphcloud.utils.wai.scene_frame import get_scene_names

## Set up basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("slurm_stage")


def check_s3_bucket_access(bucket_name):
    # Import inside function as boto3 is not a core dependency
    import boto3

    s3_client = boto3.client("s3", region_name=cfg.get("region_name"))
    try:
        # head_bucket is a lightweight way to check access
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Access to bucket '{bucket_name}' is OK.")
    except Exception:
        raise RuntimeError(
            f"No access to bucket '{bucket_name}': Please refresh your credentials.\n"
            "Please add the key, secret and token to ~/.aws/config."
        )


def pre_launch_sanity_check(main_cfg, cfg):
    wai_commit_hash = get_commit_hash(str(WAI_PROC_MAIN_PATH))
    dirty_is_bad = not cfg.get("danger_area_allow_launching_in_dirty_state", False)
    if wai_commit_hash.endswith("-dirty") and dirty_is_bad:
        raise RuntimeError(
            "Launching in dirty state is not allowed. Please commit all your "
            "changes and ideally push to remote for best backtracing of errors."
        )

    if main_cfg.stage == "copy_to_s3":
        check_s3_bucket_access(main_cfg.bucket_name)


if __name__ == "__main__":
    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  [{i}]: {arg}")

    cfg = argconf_parse()
    if cfg.get("stage") is None:
        raise ValueError(
            "Set stage via CLI, e.g. "
            "`python launch/slurm_stage.py configs/launch/scannetppv2.yaml stage=undistort`"
        )
    if cfg.get("conda_env") is None:
        raise ValueError(
            "Pass the name of your conda environment like `conda_env=pytorch`"
        )

    logger.info("Running slurm_stage using config:")
    for key, value in dict(cfg).items():
        logger.info(f"  {key}: {value}")

    stage_cfg = cfg.stages.get(cfg.stage)
    if stage_cfg is None:
        raise ValueError(f"Stage not supported: {cfg.stage}")

    launch_on_slurm = cfg.get("launch_on_slurm", False)
    # if the user wants to lock the version of the code for every stage
    # this will ensure that the current version of wai is exported as a tarbal
    # and will be used by every job. This behavior only holds if there is something
    # to launch on slurm.
    if launch_on_slurm:
        locked_cfg = cfg.get("locked", True)
    else:
        locked_cfg = False

    # No CLI overwrites as these are applied to the launch cfg
    # cfg --> launch cfg with multiple stages
    # main_cfg --> Per stage cfg as it will be passed to the stage python file
    main_cfg = argconf_parse(
        WAI_PROC_CONFIG_PATH / stage_cfg.config, cli_overwrite=False
    )

    # Sanity check before launching
    pre_launch_sanity_check(main_cfg, cfg)

    if locked_cfg and launch_on_slurm:
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(WAI_SPOD_RUNS_PATH, cfg.stage)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        shared_code_path = os.path.join(base_path, unique_id)
        shutil.copytree(WAI_PROC_MAIN_PATH, shared_code_path)
        logger.info(f"locked version of wai-processing at: {shared_code_path}")

    # Set resources using stage settings (if available)
    gpus = stage_cfg.get("gpus", cfg.gpus)
    cpus = stage_cfg.get("cpus", cfg.cpus)
    mem = stage_cfg.get("mem", cfg.mem)
    scenes_per_job = stage_cfg.get("scenes_per_job", cfg.scenes_per_job)
    spod_template = cfg.get("spod_template", "basic_cmd.sh")

    dataset_name = Path(cfg.conf).stem  # name config according to dataset
    main_cfg.root = cfg.root
    data_split = cfg.get("data_split")
    if "data_split" in main_cfg:
        main_cfg.data_split = data_split
    if cfg.get("dry_run_filter") is not None:
        logger.info(f"Prefilter for a dry run: {cfg.dry_run_filter}")
        main_cfg["scene_filters"] = main_cfg.get("scene_filters", []) + [
            cfg.dry_run_filter
        ]

    # Resolve additional CLI arguments set for this stage via the config
    # Such as scene_filters or keyword arguments
    additional_scene_filters = []
    additional_args = ""
    for cli_param in stage_cfg.get("additional_cli_params", []):
        if match := re.match(r"\+scene_filters=(.+)", cli_param):
            additional_scene_filters.append(parse_string_to_dict(match.group(1)))
        else:
            additional_args += f" {cli_param}"

    # scene_names after filtering
    if cfg.stage == "conversion":
        scene_names = import_function_from_path(
            WAI_PROC_SCRIPT_PATH / stage_cfg.script, "get_original_scene_names"
        )(main_cfg)

        # now enable filtering also on process_state (if root exists)
        if Path(main_cfg.root).exists():
            scene_names = get_scene_names(main_cfg, scene_names=scene_names)
    else:
        if additional_scene_filters:
            main_cfg.scene_filters = main_cfg.scene_filters + additional_scene_filters

        scene_names = get_scene_names(main_cfg)

    num_scenes = len(scene_names)

    logger.info(f"--- Processing {num_scenes:,} scenes ---")
    logger.debug(f"scene_names = {scene_names}")
    max_slurm_jobs = cfg.get("max_num_slurm_jobs", 20)
    # Additionaly safety measures to avoid launching 100s or 1000s of jobs on SLURM
    if num_scenes / scenes_per_job > max_slurm_jobs:
        raise RuntimeError(
            f"This would launch {ceil(num_scenes / scenes_per_job)} jobs, but only {max_slurm_jobs} allowed.\n"
            "If this is intentional you can increase the maximum number of jobs by passing the 'max_num_slurm_jobs=<your_new_max_number_of_allowed_jobs>'"
        )
    for start_idx in range(0, num_scenes, scenes_per_job):
        end_idx = min(start_idx + scenes_per_job, num_scenes)
        job_scene_names = scene_names[start_idx:end_idx]
        additional_scene_filters = [_escape_scene_names(job_scene_names)]

        common_spod_conf = {
            "run": f"{dataset_name}_{cfg.stage}_{start_idx}-{end_idx}",
            "cpus": cpus,
            "gpus": gpus,
            "mem": mem,
            "conda_env": cfg.conda_env,
            "template": spod_template,
            "nodelist": cfg.get("nodelist"),
            "args": additional_args,
        }

        if locked_cfg:
            script_path = (
                f"{shared_code_path}/wai-processing/scripts/{stage_cfg.script}"
            )
            config_path = (
                f"{shared_code_path}/wai-processing/configs/{stage_cfg.config}"
            )
        else:
            script_path = f"{WAI_PROC_SCRIPT_PATH}/{stage_cfg.script}"
            config_path = f"{WAI_PROC_CONFIG_PATH}/{stage_cfg.config}"

        cmd = (
            f"python {script_path} {config_path} "
            f"root={cfg.root} '+scene_filters={_escape_scene_names(job_scene_names)}'"
        )
        if data_split is not None:
            cmd += f" data_split={data_split}"

        spod_conf = {
            **common_spod_conf,
            "command": cmd,
        }

        if locked_cfg:
            spod_conf["working_dir"] = shared_code_path

        if launch_on_slurm:
            spod.run(**spod_conf)
        else:
            logger.info(
                f"\nWould launch with the following config:\n {json.dumps(spod_conf, indent=2)}"
            )
    if not launch_on_slurm:
        logger.info(
            "\nThis command did not launch any jobs. If the above logs of jobs that would be good look okay "
            "to you, run the command with 'launch_on_slurm=true' to schedule the jobs on SLURM."
        )
