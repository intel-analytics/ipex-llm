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
import torch
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from ..core.onnxruntime_model import ONNXRuntimeModel
import onnxruntime  # should be put behind core's import
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.pytorch import export_to_onnx, get_input_example, \
    get_forward_args
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.context_manager import generate_context_manager, BaseContextManager
from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object, \
    MetaData
from bigdl.nano.utils.common import SafePickle


class PytorchONNXRuntimeModel(ONNXRuntimeModel, AcceleratedLightningModule):
    '''
        This is the accelerated model for pytorch and onnxruntime.
        All the external API is based on Trainer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchONNXRuntimeModel will serve for all precision models.
    '''

    def __init__(self, model, input_sample=None, onnxruntime_session_options=None,
                 simplification=True, dynamic_axes=True, output_tensors=True,
                 output_metadata=None,
                 **export_kwargs):
        """
        Create a ONNX Runtime model from pytorch.

        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference
                      2. Path to ONNXRuntime saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        :param onnxruntime_session_options: A session option for onnxruntime accelerator.
        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
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
        :param output_tensors: boolean, default to True and output of the model will be Tensors.
                               If output_tensors=False, output of the ONNX model will be ndarray.
        :param output_metadata: metadata of model output, defaults to None.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        """
        # Typically, when model is int8, we use this path
        # TODO: self._forward_args should be set externally
        self.output_metadata = output_metadata
        with TemporaryDirectory() as tmpdir:
            if isinstance(model, torch.nn.Module):
                onnx_path = os.path.join(tmpdir, "tmp.onnx")
                # Typically, when model is fp32, we use this path
                export_to_onnx(model, input_sample=input_sample, onnx_path=onnx_path,
                               dynamic_axes=dynamic_axes, **export_kwargs)
                if simplification is True:
                    # simplify model
                    try:
                        from bigdl.nano.deps.onnxsim.onnxsim_api import onnx_simplify
                        onnx_simplify(onnx_path)
                    except Exception:
                        pass

                # test run to get output metadata
                with BaseContextManager():
                    forward_args = get_forward_args(model)
                    input_sample = get_input_example(model, input_sample, forward_args)
                    if isinstance(input_sample, (tuple, list)):
                        output = model(*input_sample)
                    else:
                        output = model(input_sample)
                    self.output_metadata = MetaData.construct_matadata(output)
            else:
                onnx_path = model
            AcceleratedLightningModule.__init__(self, None)
            ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)
        if onnxruntime_session_options.intra_op_num_threads > 0:
            self.thread_num = onnxruntime_session_options.intra_op_num_threads
        else:
            self.thread_num = None
        self._nano_context_manager = generate_context_manager(accelerator=None,
                                                              precision="fp32",
                                                              thread_num=self.thread_num)
        if isinstance(model, torch.nn.Module):
            # patch original model's attr to current new model
            patch_attrs_from_model_to_object(model, self)
        self.output_tensors = output_tensors

    def on_forward_start(self, inputs):
        if self.ortsess is None:
            invalidInputError(False,
                              "Please create an instance by PytorchONNXRuntimeModel()")
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_start_kwargs(self, **kwargs):
        self.cope_with_keyword_arguments(kwargs)
        return kwargs

    def on_forward_end(self, outputs):
        if self.output_tensors:
            outputs = self.numpy_to_tensors(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]
        if self.output_metadata is not None:
            outputs = MetaData.reconstruct_output(outputs, self.output_metadata)
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"onnx_path": 'onnx_saved_model.onnx',
                       "metadata_path": 'matadata.pkl',
                       "intra_op_num_threads": self.session_options.intra_op_num_threads,
                       "inter_op_num_threads": self.session_options.inter_op_num_threads,
                       "output_tensors": self.output_tensors})
        return status

    @staticmethod
    def _load(path):
        """
        Load an ONNX model for inference from directory.

        :param path: Path to model to be loaded.
        :return: PytorchONNXRuntimeModel model for ONNX Runtime inference.
        """
        status = PytorchONNXRuntimeModel._load_status(path)
        if status.get('onnx_path', None):
            onnx_path = Path(status['onnx_path'])
            invalidInputError(onnx_path.suffix == '.onnx',
                              "Path of onnx model must be with '.onnx' suffix.")
        else:
            invalidInputError(False,
                              "nano_model_meta.yml must specify 'onnx_path' for loading.")
        onnx_path = Path(path) / status['onnx_path']
        onnxruntime_session_options = onnxruntime.SessionOptions()
        if status.get('intra_op_num_threads', None):
            onnxruntime_session_options.intra_op_num_threads = \
                status.get('intra_op_num_threads', None)
        if status.get('inter_op_num_threads', None):
            onnxruntime_session_options.inter_op_num_threads = \
                status.get('inter_op_num_threads', None)
        output_tensors = status.get('output_tensors', True)
        # load meatdata
        metadata_path = status.get('metadata_path', None)
        if metadata_path is None or not metadata_path:
            output_metadata = None
        else:
            with open(path / status['metadata_path'], "rb") as f:
                output_metadata = SafePickle.load(f)
        return PytorchONNXRuntimeModel(str(onnx_path),
                                       onnxruntime_session_options=onnxruntime_session_options,
                                       output_tensors=output_tensors,
                                       output_metadata=output_metadata)

    def _save_model(self, path, compression="fp32"):
        onnx_path = Path(path) / self.status['onnx_path']
        super()._save_model(onnx_path)
        # save metadata
        with open(path / self.status['metadata_path'], "wb") as f:
            SafePickle.dump(self.output_metadata, f)
