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
from bigdl.nano.pytorch.context_manager import generate_context_manager, BaseContextManager
from bigdl.nano.utils.pytorch import get_input_example, get_forward_args, \
    patch_attrs_from_model_to_object, MetaData
from bigdl.nano.utils.common import SafePickle


class PytorchOpenVINOModel(AcceleratedLightningModule):
    def __init__(self, model, input_sample=None, precision='fp32',
                 thread_num=None, device='CPU', dynamic_axes=True,
                 logging=True, config=None, output_tensors=True,
                 shapes=None, output_metadata=None, **kwargs):
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
        :param output_tensors: boolean, default to True and output of the model will be Tensors. If
                               output_tensors=False, output of the OpenVINO model will be ndarray.
        :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',
                       '[1,3,224,224]'. This parameter affect model Parameter shape, can be
                       dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.
                       Only valid for openvino model, otherwise will be ignored.
        :param output_metadata: metadata of model output, defaults to None.
        :param **kwargs: will be passed to torch.onnx.export function or model optimizer function.
        """
        self.output_metadata = output_metadata
        ov_model_path = model
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if isinstance(model, torch.nn.Module):
                # cope with dynamic axes for GPU
                if device != 'CPU':
                    if dynamic_axes is True or (
                        isinstance(dynamic_axes, dict) and len(dynamic_axes) > 0
                    ):
                        invalidInputError("input_shape" in kwargs,
                                          "For model has dynamic axes, if you want to inference on "
                                          "non-CPU device, must define input_shape for model "
                                          "optimizer. For more details about model optimizer, you "
                                          "can see mo --help .")
                export(model, input_sample, str(tmpdir / 'tmp.xml'),
                       precision=precision, dynamic_axes=dynamic_axes,
                       logging=logging, **kwargs)
                ov_model_path = tmpdir / 'tmp.xml'

                # test run to get output metadata
                with BaseContextManager():
                    forward_args = get_forward_args(model)
                    input_sample = get_input_example(model, input_sample, forward_args)
                    if isinstance(input_sample, (list, tuple)):
                        output = model(*input_sample)
                    else:
                        output = model(input_sample)
                    self.output_metadata = MetaData.construct_matadata(output)

            self.ov_model = OpenVINOModel(ov_model_path,
                                          device=device,
                                          precision=precision,
                                          thread_num=thread_num,
                                          config=config,
                                          shapes=shapes)
            super().__init__(None)
        self._nano_context_manager = generate_context_manager(accelerator="openvino",
                                                              precision="fp32",
                                                              thread_num=thread_num)
        if isinstance(model, torch.nn.Module):
            # patch original model's attr to current new model
            patch_attrs_from_model_to_object(model, self)
        self.output_tensors = output_tensors

    def on_forward_start(self, inputs):
        self.ov_model._model_exists_or_err()
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def forward_step(self, *inputs, **kwargs):
        return self.ov_model.forward_step(*inputs, **kwargs)

    def on_forward_start_kwargs(self, **kwargs):
        self.cope_with_keyword_arguments(kwargs)
        return kwargs

    def on_forward_end(self, outputs):
        outputs = list(outputs.values())
        if self.output_tensors:
            outputs = self.numpy_to_tensors(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]
        if self.output_metadata is not None:
            outputs = MetaData.reconstruct_output(outputs, self.output_metadata)
        return outputs

    def reshape(self, shapes):
        return self.ov_model.reshape(shapes=shapes)

    @property
    def status(self):
        status = super().status
        status.update({"xml_path": 'ov_saved_model.xml',
                       "metadata_path": 'matadata.pkl',
                       "weight_path": 'ov_saved_model.bin',
                       "config": self.ov_model.final_config,
                       "device": self.ov_model._device,
                       "output_tensors": self.output_tensors,
                       })
        return status

    @property  # type: ignore
    def forward_args(self):
        return self.ov_model.forward_args

    @staticmethod
    def _load(path, device=None, cache_dir=None, shapes=None):
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
        elif "INFERENCE_NUM_THREADS" in config:
            thread_num = int(config["INFERENCE_NUM_THREADS"])
        if cache_dir is not None:
            config["CACHE_DIR"] = cache_dir
        if device is None:
            device = status.get('device', 'CPU')
        output_tensors = status.get('output_tensors', True)
        # load meatdata
        metadata_path = status.get('metadata_path', None)
        if metadata_path is None or not metadata_path:
            output_metadata = None
        else:
            with open(path / status['metadata_path'], "rb") as f:
                output_metadata = SafePickle.load(f)
        return PytorchOpenVINOModel(xml_path,
                                    config=config,
                                    thread_num=thread_num,
                                    device=device,
                                    shapes=shapes,
                                    output_tensors=output_tensors,
                                    output_metadata=output_metadata)

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
                      config=config,
                      output_tensors=self.output_tensors)
        return self

    def _save_model(self, path, compression="fp32"):
        """
        Save PytorchOpenVINOModel to local as xml and bin file

        :param path: Directory to save the model.
        """
        self.ov_model._model_exists_or_err()
        path = Path(path)
        path.mkdir(exist_ok=True)
        xml_path = path / self.status['xml_path']
        save(self.ov_model.ie_network, xml_path)
        # save metadata
        with open(path / self.status['metadata_path'], "wb") as f:
            SafePickle.dump(self.output_metadata, f)

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
