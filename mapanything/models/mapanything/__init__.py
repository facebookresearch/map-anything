# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Compatibility wrapper for legacy ``mapanything.models.mapanything`` imports."""

import morphcloud.models.morphcloud as _morphcloud_model_pkg

from morphcloud.models.morphcloud import *  # noqa: F401,F403

__all__ = list(getattr(_morphcloud_model_pkg, "__all__", []))
