# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path

WAI_PROC_MAIN_PATH = Path(__file__).parent.parent.parent
WAI_PROC_CONFIG_PATH = WAI_PROC_MAIN_PATH / "wai_processing" / "configs"
WAI_PROC_SCRIPT_PATH = WAI_PROC_MAIN_PATH / "wai_processing" / "scripts"
WAI_SPOD_RUNS_PATH = "/ai4rl/fsx/xrtech/code/spod_runs"
