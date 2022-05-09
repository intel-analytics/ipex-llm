#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
import os

try:
    from .autotsestimator import AutoTSEstimator
except ImportError:
    warnings.warn("Please install `bigdl-nano[all]` to use AutoTSEstimator")
from .tspipeline import TSPipeline

if os.getenv("LD_PRELOAD", "null") != "null":
    del os.environ["LD_PRELOAD"]
    warnings.warn("'LD_PRELOAD' has been unset to support AutoTS")
