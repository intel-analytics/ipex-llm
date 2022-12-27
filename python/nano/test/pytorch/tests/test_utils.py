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

import tempfile
import os
import torch
import time
from bigdl.nano.utils.util import spawn_new_process
from torchvision.models import resnet18


def throughput_helper(model, x):
    st = time.time()
    model(x)
    return time.time() - st, torch.get_num_threads()


def test_spawn_new_process():
    x = torch.rand(2, 3, 224, 224)
    model = resnet18(pretrained=True)

    original_thread = torch.get_num_threads()

    new_throughput_helper = spawn_new_process(throughput_helper)
    _, thread = new_throughput_helper(model, x)
    assert thread == original_thread

    _, thread = new_throughput_helper(model, x, env_var={"OMP_NUM_THREADS": "1"})
    assert thread == 1

    _, thread = new_throughput_helper(model, x, env_var={"OMP_NUM_THREADS": "2"})
    assert thread == 2
