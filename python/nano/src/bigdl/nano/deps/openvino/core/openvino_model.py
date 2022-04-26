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
from openvino.inference_engine import IECore


class OpenVINOModel:
    def __init__(self, ie_network: str):
        self.ie_network = None
        self.read_network(ie_network)
        self.exec_model = None

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

    def read_network(self, model: str):
        self.ie_network = IECore().read_network(model=model)

    def _save_model(self, path):
        """
        Save PytorchOpenVINOModel to local as xml and bin file

        :param path: Path to save the model.
        """
        path = Path(path)
        assert self.ie_network, "self.ie_network shouldn't be None."
        assert path.suffix == ".xml", "Path of openvino model must be with '.xml' suffix."
        self.ie_network.serialize(str(path))
