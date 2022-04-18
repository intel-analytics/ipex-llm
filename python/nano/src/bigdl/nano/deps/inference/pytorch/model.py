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
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
import torch

from .model_utils import get_forward_args


class AcceleratedLightningModule(LightningModuleFromTorch):
    def __init__(self, model):
        super().__init__(model)
        self.on_init_end()

    def on_init_end(self, *args):
        pass

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("This model is not trainable!")
        super().train(mode)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    def get_forward_args(self):
        return get_forward_args(self)

    @staticmethod
    def tensors_to_numpy(tensors):
        np_data = tuple(map(lambda x: x.cpu().numpy(), tensors))
        return np_data

    @staticmethod
    def numpy_to_tensors(np_array):
        tensors = tuple(map(lambda x: torch.from_numpy(x), np_array))
        if len(tensors) == 1:
            tensors = tensors[0]
        return tensors

    def save(self, path):
        raise NotImplementedError("AcceleratedLightningModule.save should be overridden.")
