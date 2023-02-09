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
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from .utils import export
import torch
from bigdl.nano.utils.common import invalidInputError
from ..core.utils import save
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.pytorch.utils import patch_attrs_from_model_to_object


class PytorchOpenVINOModel(AcceleratedLightningModule):
    def __init__(self, model, input_sample=None, precision='fp32',
                 thread_num=None, device='CPU', dynamic_axes=True,
                 logging=True, config=None, **kwargs):
        """
        Create a OpenVINO model from pytorch.

        :param model: Pytorch model to be converted to OpenVINO for inference or
                      path to Openvino saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                          defaults to 'fp32'.
        :param thread_num: a int represents how many threads(cores) is needed for
                           inference. default: None.
        :param device: A string represents the device of the inference. Default to 'CPU'.
                       'CPU', 'GPU' and 'VPUX' are supported for now.
        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model
                             will have the first dim of each Tensor input as a dynamic batch_size.
                             If dynamic_axes=False, the exported model will have the shapes of all
                             input and output tensors set to exactly match those given in
                             input_sample. To specify axes of tensors as dynamic (i.e. known only
                             at run-time), set dynamic_axes to a dict with schema:

                             | KEY (str): an input or output name. Each name must also be provided
                             | in input_names or output_names.
                             |
                             | VALUE (dict or list): If a dict, keys are axis indices and values
                             | are axis names. If a list, each element is an axis index.

                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.
        :param logging: whether to log detailed information of model conversion. default: True.
        :param config: The config to be inputted in core.compile_model.
        :param **kwargs: will be passed to torch.onnx.export function or model optimizer function.
        """
        ov_model_path = model
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if isinstance(model, torch.nn.Module):
                if device != 'CPU':
                    # workaround for dynamic shape issue on GPU/VPU plugin
                    dynamic_axes = False
                export(model, input_sample, str(tmpdir / 'tmp.xml'),
                       precision=precision, dynamic_axes=dynamic_axes,
                       logging=logging, **kwargs)
                ov_model_path = tmpdir / 'tmp.xml'

            self.ov_model = OpenVINOModel(ov_model_path,
                                          device=device,
                                          precision=precision,
                                          thread_num=thread_num,
                                          config=config)
            super().__init__(None)
        self._nano_context_manager = generate_context_manager(accelerator="openvino",
                                                              precision="fp32",
                                                              thread_num=thread_num)
        if isinstance(model, torch.nn.Module):
            # patch original model's attr to current new model
            patch_attrs_from_model_to_object(model, self)

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
        status.update({"xml_path": 'ov_saved_model.xml',
                       "weight_path": 'ov_saved_model.bin',
                       "config": self.ov_model.final_config,
                       "device": self.ov_model._device})
        return status

    @property  # type: ignore
    def forward_args(self):
        return self.ov_model.forward_args

    @staticmethod
    def _load(path, device=None):
        """
        Load an OpenVINO model for inference from directory.

        :param path: Path to model to be loaded.
        :param device: A string represents the device of the inference.
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
        thread_num = None
        config = status.get('config', {})
        if "CPU_THREADS_NUM" in config:
            thread_num = int(config["CPU_THREADS_NUM"])
        if device is None:
            device = status.get('device', 'CPU')
        return PytorchOpenVINOModel(xml_path,
                                    config=config,
                                    thread_num=thread_num,
                                    device=device)

    def pot(self,
            dataloader,
            metric=None,
            higher_better=True,
            drop_type="relative",
            maximal_drop=0.999,
            max_iter_num=1,
            n_requests=None,
            thread_num=None,
            config=None,
            sample_size=300):
        # convert torch metric/dataloader to openvino format
        if metric:
            metric = PytorchOpenVINOMetric(metric=metric, higher_better=higher_better)
        dataloader = PytorchOpenVINODataLoader(dataloader, collate_fn=self.tensors_to_numpy,
                                               original_collate_fn=dataloader.collate_fn)
        model = self.ov_model.pot(dataloader, metric=metric, drop_type=drop_type,
                                  maximal_drop=maximal_drop, max_iter_num=max_iter_num,
                                  n_requests=n_requests, sample_size=sample_size)
        # below code will re-define a new object, and original attrs will be lost.
        # return PytorchOpenVINOModel(model, thread_num=thread_num, config=config)
        self.__init__(model,
                      device=self.ov_model._device,
                      thread_num=thread_num,
                      precision='int8',
                      config=config)
        return self

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
