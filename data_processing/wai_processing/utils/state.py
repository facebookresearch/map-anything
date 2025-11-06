# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pkg_resources
import portalocker
from portalocker.exceptions import LockException

from morphcloud.utils.wai.io import _store_readable, get_processing_state

logger = logging.getLogger(__name__)


def set_processing_state(
    scene_root: Path | str,
    process_name: str,
    state: str,
    message: str = "",
):
    """
    Sets the processing state of a specific process in a scene with file locking.

    Args:
        scene_root (Path or str): The root directory of the scene.
        process_name (str): The name of the process to update.
        state (str): The new state of the process.
        message (str, optional): An optional message to associate with the process state.
            Defaults to an empty string.

    Notes:
        This function updates the processing log file ("_process_log.json") in the scene root directory.
        It adds or updates an entry for the specified process with the given state and message.
        It uses file locking to ensure thread safety when multiple processes access the file.
        Creates a backup of the existing file before overwriting it.
    """
    process_log_path = Path(scene_root) / "_process_log.json"

    # Get the current processing state
    process_log = get_processing_state(scene_root)

    # Create a new entry
    process_log[process_name] = {
        "state": state,
        "message": message,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wai_commit_hash": "",
        # NOTE: The link only works if the commit was already pushed to any branch on remote
        "wai_code_state_link": "",
        "script_path": os.path.realpath(sys.argv[0]),
    }

    # create/overwrite backup
    if process_log_path.exists():
        backup_fname = process_log_path.parent / f"{process_log_path.stem}_backup.json"
        if backup_fname.exists():
            backup_fname.unlink()
        process_log_path.rename(backup_fname)

    # Use _store_readable which already implements file locking
    _store_readable(process_log_path, process_log)


def get_commit_hash(package: str):
    try:
        if Path(package).exists():
            location = package
        else:
            # Get the distribution object for the package
            dist = pkg_resources.get_distribution(package)
            # Get the location of the package
            location = dist.location
        # Run the git command to get the commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=location)
            .decode("utf-8")
            .strip()
        )
        if (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=location)
            .decode("utf-8")
            .strip()
        ):
            commit_hash += "-dirty"
        return commit_hash
    except pkg_resources.DistributionNotFound:
        logger.error(f"Package {package} not found.")
        return None
    except subprocess.CalledProcessError:
        logger.error(f"Failed to get commit hash for package {package}.")
        return None
    except Exception as e:
        logger.error(f"Failed to get commit hash: {e}")
        return None


class SceneProcessLock:
    """
    Context manager to lock a scene while processing it and avoid race condition.
    Be careful, this works only for UNIX / NFS filesystems.
    """

    def __init__(self, scene_root, timeout=30 * 60, check_interval=60):
        self.scene_root = scene_root
        self.timeout = timeout
        self.check_interval = check_interval
        self.lockfile = self.scene_root / "_scene.lock"
        self.lock = None

    def __enter__(self):
        try:
            self.lock = portalocker.Lock(
                self.lockfile, timeout=self.timeout, check_interval=self.check_interval
            )
            logger.debug(
                f"Trying to acquire the lock for {self.lockfile} will timeout after {self.timeout} s checking every {self.check_interval} s."
            )
            self.lock.acquire()
            logger.debug(f"Lock acquired: {self.lockfile}")
        except LockException as err:
            err_message = f"Could not acquire the lock for scene {self.scene_root} on lockfile {self.lockfile} after {self.timeout} s."
            logger.error(err_message)
            raise TimeoutError(err_message) from err
        except Exception as err:
            logger.error(
                f"Unexpected error acquiring lock for {self.scene_root} on {self.lockfile}: {err}",
                exc_info=True,
            )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        self.lock = None
        logger.debug(f"Lock released: {self.lockfile}")
        return None
