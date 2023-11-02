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
from abc import ABC
import yaml
from pathlib import Path
from bigdl.nano.utils.common import invalidInputError


class AcceleratedModel(ABC):
    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        invalidInputError(False, "Method 'forward_step' is not implemented.")

    def on_forward_end(self, outputs):
        return outputs

    @staticmethod
    def tensors_to_numpy(tensors):
        invalidInputError(False, "Method 'tensors_to_numpy' is not implemented.")

    @staticmethod
    def numpy_to_tensors(np_arrays):
        invalidInputError(False, "Method 'numpy_to_tensors' is not implemented.")

    def _dump_status(self, path):
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'w') as f:
            yaml.safe_dump(self.status, f)

    def _save_model(self, path, compression="fp32"):
        """
        Save the model file to directory.

        :param path: Path to saved model. Path should be a directory.
        """
        invalidInputError(False, "Saving function is not implemented.")

    def _save(self, path, compression="fp32"):
        """
        Save the model to local file.

        :param path: Path to saved model. Path should be a directory.
        """
        path = Path(path)
        Path.mkdir(path, exist_ok=True)
        self._save_model(path, compression=compression)
        self._dump_status(path)  # don't move this before _save_model

    @property
    def status(self):
        return {"ModelType": type(self).__name__}

    @staticmethod
    def _load_status(path):
        if isinstance(path, dict):
            metadata = yaml.safe_load(path["nano_model_meta.yml"])
            path["nano_model_meta.yml"].seek(0)
        else:
            meta_path = Path(path) / "nano_model_meta.yml"
            with open(meta_path, 'r') as f:
                metadata = yaml.safe_load(f)
        return metadata

    @staticmethod
    def _load(path, model=None):
        invalidInputError(False, "Loading function is not implemented.")
