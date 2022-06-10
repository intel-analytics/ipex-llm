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
from pytorch_lightning import LightningModule
from ..model import AcceleratedModel
import torch
from pathlib import Path
from .model_utils import get_forward_args
from bigdl.nano.utils.log4Error import invalidInputError


class AcceleratedLightningModule(AcceleratedModel, LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def train(self, mode=True):
        if mode:
            invalidInputError(False, "This model is not trainable!")
        super().train(mode)

    def forward_step(self, *inputs):
        return self.model(*inputs)

    @staticmethod
    def tensors_to_numpy(tensors):
        np_data = tuple(map(lambda x: x.cpu().numpy(), tensors))
        return np_data

    @staticmethod
    def numpy_to_tensors(np_arrays):
        tensors = tuple(map(lambda x: torch.from_numpy(x), np_arrays))
        if len(tensors) == 1:
            tensors = tensors[0]
        return tensors
