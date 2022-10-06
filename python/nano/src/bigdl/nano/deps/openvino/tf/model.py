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
import tensorflow as tf
from bigdl.nano.utils.log4Error import invalidInputError
from ..core.utils import save


class KerasOpenVINOModel(AcceleratedKerasModel):
    def __init__(self, model):
        """
        Create a OpenVINO model from Keras.

        :param model: Keras model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        """
        ov_model_path = model
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            if isinstance(model, tf.keras.Model):
                export(model, str(dir / 'tmp.xml'))
                ov_model_path = dir / 'tmp.xml'
            self.ov_model = OpenVINOModel(ov_model_path)
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
        status.update({"xml_path": 'ov_saved_model.xml', "weight_path": 'ov_saved_model.bin'})
        return status

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
        return KerasOpenVINOModel(xml_path)

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
