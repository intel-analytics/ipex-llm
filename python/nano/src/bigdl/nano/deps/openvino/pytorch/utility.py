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
from ..core.utility import convert_onnx_to_xml
from ...inference.pytorch.utility import export as export_to_onnx


def export(model, input_sample=None, xml_path="model.xml"):
    '''
    Function to export pytorch model into openvino and save it to local.
    Any instance of torch.nn.Module including Lightning Module is acceptable.

    :param model: Model instance of torch.nn.module to be exported.
    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save openvino model file.
    '''
    export_to_onnx(model, input_sample, 'tmp.onnx', dynamic_axes=False)
    convert_onnx_to_xml('tmp.onnx', xml_path)
    os.remove('tmp.onnx')
