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
import os
import operator
import torch
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from ..core import version as inc_version
from neural_compressor.utils.pytorch import load
from neural_compressor.model.model import PyTorchModel
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.utils.common import compare_version


class PytorchQuantizedModel(AcceleratedLightningModule):
    def __init__(self, model, thread_num=None):
        super().__init__(model.model)
        self.quantized = model
        self.thread_num = thread_num
        self._nano_context_manager = generate_context_manager(accelerator=None,
                                                              precision="int8",
                                                              thread_num=thread_num)

    @property
    def _nargs(self):
        return -1

    @property
    def status(self):
        status = super().status
        status.update({"thread_num": self.thread_num})
        return status

    @staticmethod
    def _load(path, model, example_inputs=None):
        status = PytorchQuantizedModel._load_status(path)
        invalidInputError(
            model is not None,
            errMsg="FP32 model is required to create a quantized model."
        )
        ipex_quantization = False
        # only ipex quantization saves this file
        if isinstance(path, dict) and "best_configure.json" in path:
            ipex_quantization = True
            import intel_extension_for_pytorch as ipex
        if not isinstance(path, dict) and os.path.exists(os.path.join(path, "best_configure.json")):
            ipex_quantization = True
            import intel_extension_for_pytorch as ipex
        if ipex_quantization:
            invalidInputError(example_inputs is not None,
                              "For INC ipex quantizated model, you need to set input_sample "
                              "when loading model.")
        # INC 1.14 and 2.0 doesn't supprot quantizing pytorch-lightning module,
        # so we only quantize the internal nn.Module to fix this issue,
        # so we should load weight using internal nn.Module also
        if isinstance(path, dict):
            weights_file = path['best_model.pt']
            stat_dict = torch.jit.load(weights_file)
            if isinstance(model, LightningModule) and compare_version("neural_compressor",
                                                                      operator.ge, "2.0"):
                qmodel = PyTorchModel(load(stat_dict, model.model, example_inputs=example_inputs))
            else:
                qmodel = PyTorchModel(load(stat_dict, model, example_inputs=example_inputs))
        else:
            if isinstance(model, LightningModule) and compare_version("neural_compressor",
                                                                      operator.ge, "2.0"):
                qmodel = PyTorchModel(load(path, model.model, example_inputs=example_inputs))
            else:
                qmodel = PyTorchModel(load(path, model, example_inputs=example_inputs))
        from packaging import version
        if version.parse(inc_version) < version.parse("1.11"):
            if isinstance(path, dict):
                tune_cfg = yaml.safe_load(path['best_configure.yaml'])
                qmodel.tune_cfg = tune_cfg
            else:
                path = Path(path)
                tune_cfg_file = path / 'best_configure.yaml'
                with open(tune_cfg_file, 'r') as f:
                    tune_cfg = yaml.safe_load(f)
                    qmodel.tune_cfg = tune_cfg
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return PytorchQuantizedModel(qmodel, thread_num=thread_num)

    def _save_model(self, path, compression="fp32"):
        self.quantized.save(path)
