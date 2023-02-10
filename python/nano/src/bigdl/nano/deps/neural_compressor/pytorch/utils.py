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
from bigdl.nano.utils.common import invalidInputError


def _check_data_type(data):
    if isinstance(data, torch.Tensor):
        return
    else:
        for x in data:
            if isinstance(x, tuple):
                _check_data_type(x)
                continue
            if not isinstance(x, torch.Tensor):
                invalidInputError(False, "expect torch.Tensor here")


def _check_loader(model, loader, metric=None):
    if loader is None:
        return
    sample = next(iter(loader))
    try:
        if metric is not None:
            # only check the data when tunning
            # TODO: not only check type, but also if metric(y, yhat)
            # can return a valid result.
            # Each one must be of torch.Tensor
            _check_data_type(sample)
            if len(sample) == 2:
                x, y = sample
                if isinstance(x, torch.Tensor):
                    model(x)
                else:
                    model(*x)
            else:
                # If sample is not tuple of length 2, then it should be (x1, x2, x3, ...).
                # check if datalader yields data complied with what model requires
                # TypeError will throw if it fails.
                model(*sample)
    except (ValueError, TypeError):
        invalidInputError(False,
                          "Dataloader for quantization should yield data in format below:\n"
                          "- (tuple or Tensor, tuple or Tensor)\n"
                          "- (Tensor, Tensor, ..., Tensor). \n"
                          "Please confirm number of inputs comply with model.forward.")
