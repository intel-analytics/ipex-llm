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
from .dataloader import PytorchOpenVINODataLoader
from .metric import PytorchOpenVINOMetric
from ..core.model import OpenVINOModel
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from .utils import export
import torch
from bigdl.nano.utils.log4Error import invalidInputError
from ..core.utils import save


class PytorchOpenVINOModel(AcceleratedLightningModule):
    def __init__(self, model, input_sample=None, logging=True, **export_kwargs):
        """
        Create a OpenVINO model from pytorch.

        :param model: Pytorch model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        :param logging: whether to log detailed information of model conversion. default: True.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        """
        ov_model_path = model
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            if isinstance(model, torch.nn.Module):
                export(model, input_sample, str(dir / 'tmp.xml'), logging, **export_kwargs)
                ov_model_path = dir / 'tmp.xml'
            self.ov_model = OpenVINOModel(ov_model_path)
            super().__init__(self.ov_model)

    def on_forward_start(self, inputs):
        self.ov_model._model_exists_or_err()
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

    @property
    def forward_args(self):
        return self.ov_model.forward_args

    @staticmethod
    def _load(path):
        """
        Load an OpenVINO model for inference from directory.

        :param path: Path to model to be loaded.
        :return: PytorchOpenVINOModel model for OpenVINO inference.
        """
        status = PytorchOpenVINOModel._load_status(path)
        if status.get('xml_path', None):
            xml_path = Path(status['xml_path'])
            invalidInputError(xml_path.suffix == '.xml',
                              "Path of openvino model must be with '.xml' suffix.")
        else:
            invalidInputError(False, "nano_model_meta.yml must specify 'xml_path' for loading.")
        xml_path = Path(path) / status['xml_path']
        return PytorchOpenVINOModel(xml_path)

    def pot(self,
            dataloader,
            metric=None,
            higher_better=True,
            drop_type="relative",
            maximal_drop=0.999,
            max_iter_num=1,
            n_requests=None,
            sample_size=300):
        # convert torch metric/dataloader to openvino format
        if metric:
            metric = PytorchOpenVINOMetric(metric=metric, higher_better=higher_better)
        dataloader = PytorchOpenVINODataLoader(dataloader, collate_fn=self.tensors_to_numpy)
        model = self.ov_model.pot(dataloader, metric=metric, drop_type=drop_type,
                                  maximal_drop=maximal_drop, max_iter_num=max_iter_num,
                                  n_requests=n_requests, sample_size=sample_size)
        return PytorchOpenVINOModel(model)

    def _save_model(self, path):
        """
        Save PytorchOpenVINOModel to local as xml and bin file

        :param path: Directory to save the model.
        """
        self.ov_model._model_exists_or_err()
        path = Path(path)
        path.mkdir(exist_ok=True)
        xml_path = path / self.status['xml_path']
        save(self.ov_model.ie_network, xml_path)
