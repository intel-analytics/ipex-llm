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
from ..core.utils import convert_pb_to_xml
from pathlib import Path


def export(model, xml_path="model.xml", precision='fp32', logging=True, **kwargs):
    '''
    Function to export pytorch model into openvino and save it to local.
    Any instance of torch.nn.Module including Lightning Module is acceptable.

    :param model: Model instance of torch.nn.module to be exported.
    :param xml_path: The path to save openvino model file.
    :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                      defaults to 'fp32'.
    :param logging: whether to log detailed information of model conversion. default: True.
    :param **kwargs: will be passed to model optimizer function.
    '''
    # export a model with dynamic axes to enable IR to accept different batches and resolutions
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        pb_path = str(folder)
        model.save(str(folder))
        convert_pb_to_xml(pb_path, xml_path, precision=precision, logging=logging, **kwargs)
