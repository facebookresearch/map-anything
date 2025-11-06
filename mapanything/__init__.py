# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Compatibility package providing the legacy ``mapanything`` namespace."""

import morphcloud as _morphcloud

from morphcloud import *  # noqa: F401,F403
from morphcloud.models.morphcloud.model import MorphCloud as MapAnything

__all__ = list(getattr(_morphcloud, "__all__", []))
__all__.append("MapAnything")
