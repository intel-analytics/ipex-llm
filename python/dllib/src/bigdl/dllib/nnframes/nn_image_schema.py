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

import sys
from bigdl.dllib.utils.file_utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


def with_origin_column(dataset, imageColumn="image", originColumn="origin", bigdl_type="float"):
    return callZooFunc(bigdl_type, "withOriginColumn", dataset, imageColumn, originColumn)
