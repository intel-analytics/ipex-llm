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
from tempfile import TemporaryDirectory
from ..core.utils import convert_onnx_to_xml
from bigdl.nano.utils.pytorch import export_to_onnx
from pathlib import Path
import torch
import inspect


def export(model, input_sample=None, xml_path="model.xml",
           precision='fp32', dynamic_axes=True, logging=True,
           **kwargs):
    '''
    Function to export pytorch model into openvino and save it to local.
    Any instance of torch.nn.Module including Lightning Module is acceptable.

    :param model: Model instance of torch.nn.module to be exported.
    :param input_sample: torch.Tensor or a list for the model tracing.
    :param xml_path: The path to save openvino model file.
    :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                      defaults to 'fp32'.
    :param dynamic_axes: parameter of torch.onnx.export.
    :param logging: whether to log detailed information of model conversion. default: True.
    :param **kwargs: will be passed to torch.onnx.export function or model optimizer function.
    '''
    # export a model with dynamic axes to enable IR to accept different batches and resolutions
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        onnx_path = str(folder / 'tmp.onnx')
        # split kwargs to torch.onnx.export and mo
        export_args = inspect.getfullargspec(torch.onnx.export).args
        export_defaults = inspect.getfullargspec(torch.onnx.export).defaults
        export_args = export_args[len(export_args) - len(export_defaults):]
        export_kwargs = {}
        mo_kwargs = {}
        for key, value in kwargs.items():
            if key in export_args:
                export_kwargs[key] = value
            else:
                mo_kwargs[key] = value
        export_to_onnx(model, input_sample, onnx_path,
                       dynamic_axes=dynamic_axes, **export_kwargs)
        convert_onnx_to_xml(onnx_path, xml_path, precision=precision,
                            logging=logging, **mo_kwargs)
