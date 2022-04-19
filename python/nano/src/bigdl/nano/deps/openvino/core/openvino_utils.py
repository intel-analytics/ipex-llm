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
from openvino.inference_engine import IECore


def convert_onnx_to_xml(onnx_file_path, xml_path, batch_size=1):
    ie = IECore()
    net_onnx = ie.read_network(model=onnx_file_path)
    net_onnx.batch_size = batch_size
    net_onnx.serialize(xml_path)
