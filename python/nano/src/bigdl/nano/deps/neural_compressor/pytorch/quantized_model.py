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
from pathlib import Path
import yaml
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from ..core import version as inc_version
from neural_compressor.utils.pytorch import load
from neural_compressor.model.model import PyTorchModel


class PytorchQuantizedModel(AcceleratedLightningModule):
    def __init__(self, model):
        super().__init__(model.model)
        self.quantized = model

    @staticmethod
    def _load(path, model):
        qmodel = PyTorchModel(load(path, model))
        from packaging import version
        if version.parse(inc_version) < version.parse("1.11"):
            path = Path(path)
            tune_cfg_file = path / 'best_configure.yaml'
            with open(tune_cfg_file, 'r') as f:
                tune_cfg = yaml.safe_load(f)
                qmodel.tune_cfg = tune_cfg
        return PytorchQuantizedModel(qmodel)

    def _save_model(self, path):
        self.quantized.save(path)
