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
from ..core import version as inc_version
from bigdl.nano.utils.inference.tf.model import AcceleratedKerasModel
from bigdl.nano.utils.log4Error import invalidInputError
from neural_compressor.model.model import TensorflowModel


class KerasQuantizedModel(AcceleratedKerasModel):

    def __init__(self, model):
        super().__init__(model)
        self._input = model.input_tensor
        self._output = model.output_tensor
        self._sess = model.sess

    def on_forward_start(self, inputs):
        return self.tensors_to_numpy(inputs)

    def forward_step(self, *inputs):
        input_dict = dict(zip(self._input, inputs))
        out = self._sess.run(self._output, feed_dict=input_dict)
        return out

    def on_forward_end(self, outputs):
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"ModelType": type(self.target_obj).__name__})
        return status

    def _save_model(self, path):
        self.model.save(path)

    @staticmethod
    def _load(path, model=None):
        status = KerasQuantizedModel._load_status(path)
        invalidInputError(
            model is not None,
            errMsg="FP32 model is required to create a quantized model."
        )
        qmodel = TensorflowModel("saved_model", str(path))
        from packaging import version
        if version.parse(inc_version) < version.parse("1.11"):
            path = Path(path)
            tune_cfg_file = path / 'best_configure.yaml'
            with open(tune_cfg_file, 'r') as f:
                tune_cfg = yaml.safe_load(f)
                qmodel.tune_cfg = tune_cfg
        return KerasQuantizedModel(qmodel)
