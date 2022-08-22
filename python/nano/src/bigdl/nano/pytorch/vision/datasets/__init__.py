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

from bigdl.nano.pytorch.utils import TORCHVISION_VERSION_LESS_1_12
from torchvision.datasets import *
del ImageFolder
if not TORCHVISION_VERSION_LESS_1_12:
    del OxfordIIITPet
    from .oxfordpet_datasets import OxfordIIITPet

from .datasets import ImageFolder, SegmentationImageFolder
