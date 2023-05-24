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


import operator

from bigdl.nano.utils.common import compare_version


TORCH_VERSION_LESS_1_10 = compare_version("torch", operator.lt, "1.10")
TORCH_VERSION_LESS_1_11 = compare_version("torch", operator.lt, "1.11")
TORCH_VERSION_LESS_1_12 = compare_version("torch", operator.lt, "1.12")
TORCH_VERSION_LESS_1_13 = compare_version("torch", operator.lt, "1.13")
TORCH_VERSION_LESS_2_0 = compare_version("torch", operator.lt, "2.0")
TORCHVISION_VERSION_LESS_1_12 = compare_version("torchvision", operator.lt, "0.12.0")
TORCHVISION_VERSION_LESS_1_14 = compare_version("torchvision", operator.lt, "0.14.0")
