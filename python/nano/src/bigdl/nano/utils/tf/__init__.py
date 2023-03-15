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


from .version import KERAS_VERSION_LESS_2_9
from .version import KERAS_VERSION_LESS_2_10

from .attributes import _ModuleWrapper
from .attributes import patch_attrs
from .attributes import patch_compiled
from .attributes import patch_compiled_and_attrs

from .backend import MultiprocessingBackend

from .preprocess import try_fake_inference

from .data import convert
from .data import convert_all
from .data import numpy_to_tensors
from .data import tensors_to_numpy
