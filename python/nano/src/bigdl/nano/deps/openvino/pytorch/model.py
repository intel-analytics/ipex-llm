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
from typing import List, Union  # for typehint
from .dataloader import PytorchOpenVINODataLoader
from .metric import PytorchOpenVINOMetric
from ..core.model import OpenVINOModel
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from .utils import export
import torch
from bigdl.nano.utils.log4Error import invalidInputError
from ..core.utils import save
from torch.utils.data.dataloader import DataLoader


class PytorchOpenVINOModel(AcceleratedLightningModule):
    def __init__(self, model, input_sample=None, thread_num=None,
                 logging=True, **export_kwargs):
        """
        Create a OpenVINO model from pytorch.

        :param model: Pytorch model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        :param thread_num: a int represents how many threads(cores) is needed for
                           inference. default: None.
        :param logging: whether to log detailed information of model conversion. default: True.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        """
        ov_model_path = model
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            if isinstance(model, torch.nn.Module):
                export(model, input_sample, str(dir / 'tmp.xml'), logging, **export_kwargs)
                ov_model_path = dir / 'tmp.xml'

            self.ov_model = OpenVINOModel(ov_model_path, thread_num=thread_num)
            super().__init__(None)

    def on_forward_start(self, inputs):
        self.ov_model._model_exists_or_err()
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def forward_step(self, *inputs):
        return self.ov_model.forward_step(*inputs)

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs.values())
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"xml_path": 'ov_saved_model.xml', "weight_path": 'ov_saved_model.bin'})
        return status

    @property  # type: ignore
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

    def async_predict(self,
                      input_data: Union[DataLoader, List[torch.Tensor], List[List[torch.Tensor]]],
                      num_requests: int = 0) -> List[torch.Tensor]:
        """
        Perfrom model inference using async mode.

        :param input_data: Input data to be inferenced.
                           Users can put multiple input data in a list or
                           put all data in a DataLoader to infer and get results of all input data.
                           Can be torch.utils.data.dataloader.DataLoader and
                           List[torch.Tensor] or
                           List[List[torch.Tensor]] if the model has multiple inputs.
                           If input_data is a DataLoader object,the format in DataLoader should be
                           (x1, x2, ..., xn, y).
        :param num_requests: Numer of requests in the asynchronous infer requests pool.
                             Each element in input_data will be bound to an idle async
                             infer request in the pool to do inference.
                             Defaults to 0.
                             If 0, it will be set automatically to the optimal number.

        :return: A List of torch.Tensor containing result of each input
        """
        if isinstance(input_data, DataLoader):
            # input_data is a DataLoader, retrieve every batch and put them in a list.
            input_list = []
            for data in input_data:
                all_inputs = list(data)[:-1]
                input_list.append(all_inputs)
        else:
            # input_data is list of torch.Tensor
            input_list = input_data

        if isinstance(input_list[0], list) and len(input_list[0]) > 1:
            # multiple inputs
            for i, inputs in enumerate(input_list):
                input_list[i] = list(self.on_forward_start(inputs))
        else:
            # single input

            # if single input in dataloader, the format is [[batch1], [batch2]]
            # after being converted to list
            # convert to [batch1, batch2] here.
            if isinstance(input_list[0], list):
                input_list = [x[0] for x in input_list]

            input_list = list(self.on_forward_start(input_list))

        results = self.ov_model.async_predict(input_list, num_requests)
        results = self.numpy_to_tensors(results)
        return results
