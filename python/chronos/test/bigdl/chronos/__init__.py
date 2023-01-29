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
import pytest

# framework
op_torch = pytest.mark.torch
op_tf2 = pytest.mark.tf2

# distribution and automl
op_automl = pytest.mark.automl
op_distributed = pytest.mark.distributed

# other mark
op_inference = pytest.mark.inference
# The difference set in all, excluding dependencies of installation options such as torch and tf2.
op_diff_set_all = pytest.mark.diff_set_all

