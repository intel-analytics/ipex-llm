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
import torch


def _check_data_type(data):
    if isinstance(data, torch.Tensor):
        return
    else:
        for x in data:
            assert isinstance(x, torch.Tensor), ValueError


def _check_loader(loader):
    if loader is None:
        return
    sample = next(iter(loader))
    try:
        x, y = sample
        _check_data_type(x)
        _check_data_type(y)
    except ValueError:
        raise ValueError(
            "Dataloader for quantization in INC should yield data in format below:\n"
            "- torch.Tensor, torch.Tensor\n"
            "- Tuple(torch.Tensor), Tuple(torch.Tensor)\n"
        )


def check_loaders(loaders):
    if isinstance(loaders, list):
        for loader in loaders:
            _check_loader(loader)
    else:
        _check_loader(loaders)
