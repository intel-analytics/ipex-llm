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
import torch
import yaml
from pathlib import Path
from .model_utils import get_forward_args
from bigdl.nano.utils.log4Error import invalidInputError


class AcceleratedLightningModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def train(self, mode=True):
        if mode:
            invalidInputError(False, "This model is not trainable!")
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
    def numpy_to_tensors(np_arrays):
        tensors = tuple(map(lambda x: torch.from_numpy(x), np_arrays))
        if len(tensors) == 1:
            tensors = tensors[0]
        return tensors

    def _dump_status(self, path):
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'w') as f:
            yaml.safe_dump(self.status, f)

    def _save_model(self, path):
        """
        Save the model file to directory.

        :param path: Path to saved model. Path should be a directory.
        """
        invalidInputError(False, "Saving function is not implemented.")

    def _save(self, path):
        """
        Save the model to local file.

        :param path: Path to saved model. Path should be a directory.
        """
        path = Path(path)
        Path.mkdir(path, exist_ok=True)
        self._dump_status(path)
        self._save_model(path)

    @property
    def status(self):
        return {"ModelType": type(self).__name__}

    @staticmethod
    def _load_status(path):
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        return metadata

    @staticmethod
    def _load(path, model=None):
        invalidInputError(False, "Loading function is not implemented.")
