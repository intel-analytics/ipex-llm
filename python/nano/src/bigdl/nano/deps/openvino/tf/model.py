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
from tempfile import TemporaryDirectory
from ..core.model import OpenVINOModel
from bigdl.nano.tf.model import AcceleratedKerasModel
from .utils import export
from .dataloader import KerasOpenVINODataLoader
from .metric import KerasOpenVINOMetric
import tensorflow as tf
from bigdl.nano.utils.common import invalidInputError
from ..core.utils import save
import pickle
import os


class KerasOpenVINOModel(AcceleratedKerasModel):
    def __init__(self, model, input_spec=None, precision='fp32',
                 thread_num=None, device='CPU', config=None,
                 logging=True, **kwargs):
        """
        Create a OpenVINO model from Keras.

        :param model: Keras model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining
                           the shape/dtype of the input
        :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                          defaults to 'fp32'.
        :param thread_num: a int represents how many threads(cores) is needed for
                    inference. default: None.
        :param device: A string represents the device of the inference. Default to 'CPU'.
                       'CPU', 'GPU' and 'VPUX' are supported for now.
        :param config: The config to be inputted in core.compile_model.
                       inference. default: None.
        :param logging: whether to log detailed information of model conversion.
                        default: True.
        :param **kwargs: will be passed to model optimizer function.
        """
        ov_model_path = model
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            if isinstance(model, tf.keras.Model):
                saved_model_input_spec_set = model._saved_model_inputs_spec is not None
                if not model.built and not saved_model_input_spec_set:
                    invalidInputError(input_spec is not None,
                                      "`input_spec` cannot be None when passing unbuilt model.")
                    # model cannot be saved either because the input shape is not available
                    # or because the forward pass of the model is not defined
                    if isinstance(input_spec, (tuple, list)):
                        input_shape = (i.shape for i in input_spec)
                    else:
                        input_shape = input_spec.shape
                    self._output_shape = model.compute_output_shape(input_shape)
                else:
                    self._output_shape = model.output_shape

                export(model, str(dir / 'tmp.xml'),
                       precision=precision,
                       logging=logging,
                       **kwargs)
                ov_model_path = dir / 'tmp.xml'
            self.ov_model = OpenVINOModel(ov_model_path,
                                          device=device,
                                          precision=precision,
                                          thread_num=thread_num,
                                          config=config)
            super().__init__(None)

    def forward_step(self, *inputs):
        return self.ov_model.forward_step(*inputs)

    def on_forward_start(self, inputs):
        self.ov_model._model_exists_or_err()
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = tuple(outputs.values())
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"xml_path": 'ov_saved_model.xml',
                       "attr_path": "ov_saved_model_attr.pkl",
                       "compile_path": "ov_saved_model_compile.pkl",
                       "weight_path": 'ov_saved_model.bin',
                       "config": self.ov_model.final_config,
                       "device": self.ov_model._device})
        return status

    def pot(self,
            x,
            y,
            metric=None,
            higher_better=True,
            drop_type="relative",
            maximal_drop=0.999,
            max_iter_num=1,
            n_requests=None,
            config=None,
            sample_size=300,
            thread_num=None):
        if metric:
            metric = KerasOpenVINOMetric(metric=metric, higher_better=higher_better)
        dataloader = KerasOpenVINODataLoader(x, y, collate_fn=self.tensors_to_numpy)
        model = self.ov_model.pot(dataloader, metric=metric, drop_type=drop_type,
                                  maximal_drop=maximal_drop, max_iter_num=max_iter_num,
                                  n_requests=n_requests, sample_size=sample_size)
        q_model = KerasOpenVINOModel(model, precision='int8',
                                     config=config, thread_num=thread_num,
                                     device=self.ov_model._device)
        q_model._output_shape = self._output_shape
        return q_model

    @staticmethod
    def _load(path, device=None):
        """
        Load an OpenVINO model for inference from directory.

        :param path: Path to model to be loaded.
        :param device: A string represents the device of the inference.
        :return: KerasOpenVINOModel model for OpenVINO inference.
        """
        status = KerasOpenVINOModel._load_status(path)
        if status.get('xml_path', None):
            xml_path = Path(status['xml_path'])
            invalidInputError(xml_path.suffix == '.xml',
                              "Path of openvino model must be with '.xml' suffix.")
        else:
            invalidInputError(False, "nano_model_meta.yml must specify 'xml_path' for loading.")
        xml_path = Path(path) / status['xml_path']
        thread_num = None
        config = status.get('config', {})
        if "CPU_THREADS_NUM" in config:
            thread_num = int(config["CPU_THREADS_NUM"])
        if device is None:
            device = status.get('device', 'CPU')
        model = KerasOpenVINOModel(xml_path,
                                   config=status['config'],
                                   thread_num=thread_num,
                                   device=device)
        with open(Path(path) / status['attr_path'], "rb") as f:
            attrs = pickle.load(f)
        for attr_name, attr_value in attrs.items():
            setattr(model, attr_name, attr_value)
        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = pickle.load(f)
                model.compile(**kwargs)
        return model

    def _save_model(self, path):
        """
        Save KerasOpenVINOModel to local as xml and bin file

        :param path: Directory to save the model.
        """
        self.ov_model._model_exists_or_err()
        path = Path(path)
        path.mkdir(exist_ok=True)
        xml_path = path / self.status['xml_path']
        save(self.ov_model.ie_network, xml_path)
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
