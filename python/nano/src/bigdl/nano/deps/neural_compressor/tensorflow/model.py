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
from bigdl.nano.utils.common import SafePickle
import yaml
from pathlib import Path
from typing import Sequence, Any, Union, Dict

from neural_compressor.model.model import TensorflowModel

from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.tf import convert_all
from bigdl.nano.tf.model import KerasOptimizedModel

from ..core import version as inc_version


class KerasQuantizedModel(KerasOptimizedModel):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._input = model.input_tensor
        self._output = model.output_tensor
        self._sess = model.sess

    def preprocess(self, args: Sequence[Any], kwargs: Dict[str, Any]):
        return convert_all(args, "numpy")

    def forward(self, inputs: Union[Sequence[Any], Dict[str, Any]]):
        input_dict = dict(zip(self._input, inputs))
        outputs = self._sess.run(self._output, feed_dict=input_dict)
        return outputs

    def postprocess(self, outputs: Sequence[Any]):
        outputs = convert_all(outputs, "tf")
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"compile_path": "inc_saved_model_compile.pkl"})
        return status

    def _save(self, path, compression="fp32"):
        path = Path(path)
        path.mkdir(exist_ok=True)

        self.model.save(path)
        self._dump_status(path)

        # save compile attr
        if self._is_compiled:
            kwargs = {"run_eagerly": self._run_eagerly,
                      "steps_per_execution": int(self._steps_per_execution)}
            if self.compiled_loss is not None:
                kwargs["loss"] = self.compiled_loss._user_losses
                kwargs["loss_weights"] = self.compiled_loss._user_loss_weights
            if self.compiled_metrics is not None:
                user_metric = self.compiled_metrics._user_metrics
                if user_metric is not None:
                    if isinstance(user_metric, (list, tuple)):
                        kwargs["metrics"] = [m._name for m in user_metric]
                    else:
                        kwargs["metrics"] = user_metric._name
                weighted_metrics = self.compiled_metrics._user_weighted_metrics
                if weighted_metrics is not None:
                    if isinstance(weighted_metrics, (list, str)):
                        kwargs["weighted_metrics"] = [m._name for m in weighted_metrics]
                    else:
                        kwargs["weighted_metrics"] = weighted_metrics._name
            with open(path / self.status['compile_path'], "wb") as f:
                SafePickle.dump(kwargs, f)

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
        model = KerasQuantizedModel(qmodel)

        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = SafePickle.load(f)
                model.compile(**kwargs)
        return model
