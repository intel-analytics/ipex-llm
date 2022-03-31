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
try:
    from openvino.inference_engine import IECore
except ImportError:
    raise ImportError("To enable openvino inference, you need to install it by:\n"
                      "\t\t pip install openvino-dev")


class BaseOpenVINOInference:
    def __init__(self, ie_network=None):
        self.ie_network = ie_network
        self.exec_model = None
        self.example_input_array = None

    def forward_step(self, *inputs):
        self.compile_executable(inputs)
        inputs = dict(zip(self.exec_model.input_info, inputs))
        return self.exec_model.infer(inputs)

    def compile_executable(self, inputs):
        input_batch_size = inputs[0].shape[0]
        if self.exec_model and input_batch_size == self.exec_model.input_info[0].shape[0]:
            return
        else:
            assert self.ie_network, "self.ie_network shouldn't be None."
            self.ie_network.batch_size = input_batch_size
            self.exec_model = IECore().load_network(network=self.ie_network, device_name='CPU')

    def reset(self, model):
        self.ie_network = None
        self.exec_model = None
        model.ov_infer_engine = None

    def read_network(self, model: str):
        self.ie_network = IECore().read_network(model=model)


def convert_onnx_to_xml(onnx_file_path, xml_path, batch_size=1):
    ie = IECore()
    net_onnx = ie.read_network(model=onnx_file_path)
    net_onnx.batch_size = batch_size
    net_onnx.serialize(xml_path)
