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
from bigdl.nano.utils.inference.tf.model import AcceleratedKerasModel
from .utils import export
from .dataloader import KerasOpenVINODataLoader
from .metric import KerasOpenVINOMetric
import tensorflow as tf
from bigdl.nano.utils.log4Error import invalidInputError
from ..core.utils import save


class KerasOpenVINOModel(AcceleratedKerasModel):
    def __init__(self, model, thread_num=None, config=None, logging=True):
        """
        Create a OpenVINO model from Keras.

        :param model: Keras model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param thread_num: a int represents how many threads(cores) is needed for
                    inference. default: None.
        :param config: The config to be inputted in core.compile_model.
                       inference. default: None.
        :param logging: whether to log detailed information of model conversion.
                        default: True.
        """
        ov_model_path = model
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            if isinstance(model, tf.keras.Model):
                export(model, str(dir / 'tmp.xml'), logging=logging)
                ov_model_path = dir / 'tmp.xml'
            self.ov_model = OpenVINOModel(ov_model_path,
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
                       "weight_path": 'ov_saved_model.bin',
                       "config": self.ov_model.final_config})
        return status

    def pot(self,
            dataset,
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
        dataloader = KerasOpenVINODataLoader(dataset, collate_fn=self.tensors_to_numpy)
        model = self.ov_model.pot(dataloader, metric=metric, drop_type=drop_type,
                                  maximal_drop=maximal_drop, max_iter_num=max_iter_num,
                                  n_requests=n_requests, sample_size=sample_size)
        return KerasOpenVINOModel(model, config=config, thread_num=thread_num)

    @staticmethod
    def _load(path):
        """
        Load an OpenVINO model for inference from directory.

        :param path: Path to model to be loaded.
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
        return KerasOpenVINOModel(xml_path, config=status['config'])

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
