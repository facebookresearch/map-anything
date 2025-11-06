# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Legacy entry point for loading pretrained MapAnything checkpoints."""

from morphcloud.models.morphcloud.model import MorphCloud, MorphCloud as MapAnything

__all__ = ["MapAnything", "MorphCloud"]
