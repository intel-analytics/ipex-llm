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
import tensorflow as tf
from bigdl.nano.utils.common import compare_version


KERAS_VERSION_LESS_2_9 = compare_version("keras", operator.lt, "2.9")
KERAS_VERSION_LESS_2_10 = compare_version("keras", operator.lt, "2.10")
