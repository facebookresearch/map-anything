# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path

import numpy as np
import pytest

from mapanything.utils.hf_utils.hf_helpers import load_prediction_archive

REPO_ROOT = Path(__file__).resolve().parents[1]
GRADIO_APP_PATH = REPO_ROOT / "scripts" / "gradio_app.py"


def _write_sentinel(path):
    Path(path).write_text("unsafe deserialization", encoding="utf-8")


class _MaliciousValue:
    def __init__(self, sentinel_path):
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return _write_sentinel, (self.sentinel_path,)


def _is_gradio_call(node, method_name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gr"
        and node.func.attr == method_name
    )


def test_load_prediction_archive_round_trips_numeric_arrays(tmp_path):
    archive_path = tmp_path / "predictions.npz"
    expected = {
        "depth": np.arange(12, dtype=np.float32).reshape(3, 4),
        "final_mask": np.array([[True, False], [False, True]]),
    }
    np.savez(archive_path, **expected)

    loaded = load_prediction_archive(archive_path)

    assert loaded.keys() == expected.keys()
    for key, expected_value in expected.items():
        np.testing.assert_array_equal(loaded[key], expected_value)


def test_load_prediction_archive_rejects_object_arrays_without_side_effect(tmp_path):
    archive_path = tmp_path / "malicious_predictions.npz"
    sentinel_path = tmp_path / "deserialization_sentinel"
    payload = np.array([_MaliciousValue(str(sentinel_path))], dtype=object)
    np.savez(archive_path, payload=payload)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_prediction_archive(archive_path)

    assert not sentinel_path.exists()


def test_gradio_app_keeps_target_directory_in_session_state():
    tree = ast.parse(GRADIO_APP_PATH.read_text(encoding="utf-8"))
    target_dir_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "target_dir_state"
            for target in node.targets
        )
    ]

    assert len(target_dir_assignments) == 1
    state_call = target_dir_assignments[0].value
    assert _is_gradio_call(state_call, "State")
    value_keyword = next(
        keyword for keyword in state_call.keywords if keyword.arg == "value"
    )
    assert isinstance(value_keyword.value, ast.Constant)
    assert value_keyword.value.value is None
    assert not any(
        isinstance(node, ast.Name) and node.id == "target_dir_output"
        for node in ast.walk(tree)
    )


def test_gradio_app_disables_public_share_links():
    tree = ast.parse(GRADIO_APP_PATH.read_text(encoding="utf-8"))
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]

    assert launch_calls
    for launch_call in launch_calls:
        share_keyword = next(
            keyword for keyword in launch_call.keywords if keyword.arg == "share"
        )
        assert isinstance(share_keyword.value, ast.Constant)
        assert share_keyword.value.value is False
