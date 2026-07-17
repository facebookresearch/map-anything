# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import json

import pytest
import torch.nn as nn

from mapanything.models.mapanything.model import MapAnything


@pytest.mark.parametrize(
    ("encoder_config", "expected_torch_hub_pretrained"),
    [
        ({"encoder_str": "dinov2", "uses_torch_hub": True}, False),
        (
            {
                "encoder_str": "dinov2",
                "uses_torch_hub": True,
                "torch_hub_pretrained": True,
            },
            False,
        ),
        ({"encoder_str": "radio", "uses_torch_hub": True}, None),
    ],
)
def test_from_pretrained_configures_encoder_weight_loading(
    monkeypatch, tmp_path, encoder_config, expected_torch_hub_pretrained
):
    captured_kwargs = {}
    original_encoder_config = copy.deepcopy(encoder_config)
    (tmp_path / "config.json").write_text(
        json.dumps({"encoder_config": encoder_config}),
        encoding="utf-8",
    )

    def capture_init(self, **kwargs):
        nn.Module.__init__(self)
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(MapAnything, "__init__", capture_init)
    monkeypatch.setattr(
        MapAnything,
        "_load_as_safetensor",
        staticmethod(lambda model, *_args, **_kwargs: model),
    )

    MapAnything.from_pretrained(
        tmp_path,
        local_files_only=True,
    )

    assert (
        captured_kwargs["encoder_config"].get("torch_hub_pretrained")
        is expected_torch_hub_pretrained
    )
    assert encoder_config == original_encoder_config
