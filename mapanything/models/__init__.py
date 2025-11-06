# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Legacy compatibility layer for ``mapanything.models`` imports."""

import morphcloud.models as _morphcloud_models

from morphcloud.models import *  # noqa: F401,F403

__all__ = list(getattr(_morphcloud_models, "__all__", []))
