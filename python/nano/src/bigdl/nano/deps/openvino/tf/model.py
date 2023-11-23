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
from pathlib import Path
from typing import Sequence, Any, Union, Dict
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.tf import try_fake_inference
from bigdl.nano.utils.tf import convert_all, tensors_to_numpy
from bigdl.nano.tf.model import KerasOptimizedModel

from ..core.model import OpenVINOModel
from ..core.utils import save
from .utils import export
from .dataloader import KerasOpenVINODataLoader
from .metric import KerasOpenVINOMetric


class KerasOpenVINOModel(KerasOptimizedModel):
    def __init__(self, model, input_spec=None, precision='fp32',
                 thread_num=None, device='CPU', config=None,
                 logging=True, shapes=None, **kwargs):
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
        :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',
                       '[1,3,224,224]'. This parameter affect model Parameter shape, can be
                       dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.
                       Only valid for openvino model, otherwise will be ignored.
        :param **kwargs: will be passed to model optimizer function.
        """
        super().__init__()
        ov_model_path = model
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            if isinstance(model, tf.keras.Model):
                self._mode = "arg"
                self._arg_names = []
                if isinstance(input_spec, dict):
                    self._arg_names = list(input_spec.keys())
                    input_spec = [input_spec[name] for name in self._arg_names]
                    self._mode = "kwarg"
                    kwargs["input"] = ','.join(self._arg_names)
                try_fake_inference(model, input_spec)
                export(model, str(tmp_dir / 'tmp.xml'),
                       precision=precision,
                       logging=logging,
                       **kwargs)
                ov_model_path = tmp_dir / 'tmp.xml'
            self.ov_model = OpenVINOModel(ov_model_path,
                                          device=device,
                                          precision=precision,
                                          thread_num=thread_num,
                                          config=config,
                                          shapes=shapes)

    def preprocess(self, args: Sequence[Any], kwargs: Dict[str, Any]):
        self.ov_model._model_exists_or_err()
        # todo: We should perform dtype conversion based on the
        # dtype of the arguments that the model expects
        inputs = args if self._mode == "arg" else [kwargs[name] for name in self._arg_names]
        inputs = convert_all(inputs, types="numpy", dtypes=np.float32)
        return inputs

    def forward(self, inputs: Sequence[Any]):   # type: ignore[overrite]
        return self.ov_model.forward_step(*inputs)

    def postprocess(self, outputs: Sequence[Any]):
        outputs = tuple(outputs.values())
        # todo: we should perform dtype conversion based on the
        # dtype of the outputs of the original keras model
        outputs = convert_all(outputs, types="tf", dtypes=tf.float32)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"attr_path": "ov_saved_model_attr.pkl",
                       "xml_path": 'ov_saved_model.xml',
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
        # todo: replace `KerasOpenVINODataLoader` and `tensors_to_numpy`
        dataloader = KerasOpenVINODataLoader(x, y,
                                             collate_fn=tensors_to_numpy)
        model = self.ov_model.pot(dataloader, metric=metric, drop_type=drop_type,
                                  maximal_drop=maximal_drop, max_iter_num=max_iter_num,
                                  n_requests=n_requests, sample_size=sample_size)
        q_model = KerasOpenVINOModel(model, precision='int8',
                                     config=config, thread_num=thread_num,
                                     device=self.ov_model._device)
        q_model._mode = self._mode
        q_model._arg_names = self._arg_names
        return q_model

    @staticmethod
    def _load(path, device=None, cache_dir=None, shapes=None):
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
        elif "INFERENCE_NUM_THREADS" in config:
            thread_num = int(config["INFERENCE_NUM_THREADS"])
        if cache_dir is not None:
            config["CACHE_DIR"] = cache_dir
        if device is None:
            device = status.get('device', 'CPU')
        model = KerasOpenVINOModel(xml_path,
                                   config=status['config'],
                                   thread_num=thread_num,
                                   device=device,
                                   shapes=shapes)
        with open(Path(path) / status['attr_path'], "rb") as f:
            attrs = SafePickle.load(f)
        for attr_name, attr_value in attrs.items():
            setattr(model, attr_name, attr_value)
        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = SafePickle.load(f)
                model.compile(**kwargs)
        return model

    def _save(self, path, compression="fp32"):
        """
        Save KerasOpenVINOModel to local as xml and bin file

        :param path: Directory to save the model.
        """
        self.ov_model._model_exists_or_err()

        path = Path(path)
        path.mkdir(exist_ok=True)
        self._dump_status(path)

        xml_path = path / self.status['xml_path']
        save(self.ov_model.ie_network, xml_path)

        attrs = {"_mode": self._mode,
                 "_arg_names": self._arg_names}
        with open(path / self.status["attr_path"], "wb") as f:
            SafePickle.dump(attrs, f)

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
                SafePickle.dump(kwargs, f)
