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
from bigdl.nano.tf.model import AcceleratedKerasModel
from bigdl.nano.utils.common import invalidInputError
from neural_compressor.model.model import TensorflowModel
import pickle
import os


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
        status.update({"attr_path": "inc_saved_model_attr.pkl",
                       "compile_path": "inc_saved_model_compile.pkl"})
        return status

    def _save_model(self, path):
        self.model.save(path)
        # save normal attrs
        attrs = {"_output_shape": self._output_shape}
        with open(Path(path) / self.status['attr_path'], "wb") as f:
            pickle.dump(attrs, f)
        # save compile attr
        if self._is_compiled:
            kwargs = {"run_eagerly": self._run_eagerly,
                      "steps_per_execution": int(self._steps_per_execution)}
            if self.compiled_loss is not None:
                kwargs["loss"] = self.compiled_loss._user_losses
                kwargs["loss_weights"] = self.compiled_loss._user_loss_weights
            if self.compiled_metrics is not None:
                user_metric = self.compiled_metrics._user_metrics
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
            with open(Path(path) / self.status['compile_path'], "wb") as f:
                pickle.dump(kwargs, f)

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
        with open(Path(path) / status['attr_path'], "rb") as f:
            attrs = pickle.load(f)
        for attr_name, attr_value in attrs.items():
            setattr(model, attr_name, attr_value)
        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = pickle.load(f)
                model.compile(**kwargs)
        return model
