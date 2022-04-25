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
from ..core.openvino_model import OpenVINOModel
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from .pytorch_openvino_utils import export
import torch


class PytorchOpenVINOModel(OpenVINOModel, AcceleratedLightningModule):
    def __init__(self, model, input_sample=None):
        """
        Create a OpenVINO model from pytorch.

        :param model: Pytorch model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        """
        ov_model_path = model
        if isinstance(model, torch.nn.Module):
            export(model, input_sample, 'tmp.xml')
            ov_model_path = 'tmp.xml'
        OpenVINOModel.__init__(self, ov_model_path)
        AcceleratedLightningModule.__init__(self, None)
        if os.path.exists('tmp.xml'):
            os.remove('tmp.xml')

    def on_forward_start(self, inputs):
        if self.ie_network is None:
            raise RuntimeError(
                "Please create an instance by PytorchOpenVINOModel() or PytorchOpenVINOModel.load()"
            )
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs.values())
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"xml_path": 'ov_saved_model.xml', "weight_path": 'ov_saved_model.bin'})
        return status

    @staticmethod
    def load(path):
        """
        Load an OpenVINO model for inference from directory.

        :param path: Path to model to be loaded.
        :return: PytorchOpenVINOModel model for OpenVINO inference.
        """
        status = PytorchOpenVINOModel.load_status(path)
        assert status.get('xml_path', None), "xml_path must not be None for loading."
        assert status['xml_path'].split('.')[-1] == "xml", \
            "Path of openvino model must be with '.xml' suffix."
        xml_path = "{}/{}".format(path, status['xml_path'])
        return PytorchOpenVINOModel(xml_path)

    def save(self, path):
        """
        Save the model to local file.

        :param path: Path to saved model. Path should be a directory.
        """
        os.makedirs(path, exist_ok=True)
        self.dump_status(path)
        super().save("{}/{}".format(path, self.status['xml_path']))
