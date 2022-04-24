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
import os

import yaml
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from neural_compressor.utils.pytorch import load
from neural_compressor.model.model import PyTorchModel


class PytorchQuantizedModel(AcceleratedLightningModule):
    def __init__(self, model):
        super().__init__(model.model)
        self.quantized = model

    def save(self, path):
        self.quantized.save(path)
        super().save(path)

    def load_state_dict(self, state_dict):
        load(state_dict, self.quantized._model)

    def state_dict(self):
        try:
            stat_dict = self.quantized.model.state_dict()
            stat_dict['best_configure'] = self.quantized.tune_cfg
        except IOError as e:
            raise IOError("Fail to dump configure and weights due to {}.".format(e))
        return stat_dict

    @staticmethod
    def load(path, model):
        qmodel = PyTorchModel(load(path, model))
        tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(path)),
                                    'best_configure.yaml')
        with open(tune_cfg_file, 'r') as f:
            tune_cfg = yaml.safe_load(f)
            qmodel.tune_cfg = tune_cfg
        return QuantizedModel(qmodel)
