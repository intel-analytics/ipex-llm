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
__all__ = ["Trainer", "TorchNano", "InferenceOptimizer"]


# unset the KMP_INIT_AT_FORK
# which will cause significant slow down in multiprocessing training
import os
if 'KMP_INIT_AT_FORK' in os.environ:
    del os.environ['KMP_INIT_AT_FORK']
from .dispatcher import patch_torch, unpatch_torch
from bigdl.nano.pytorch.inference import InferenceOptimizer
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.torch_nano import TorchNano
from bigdl.nano.pytorch.torch_nano import nano

