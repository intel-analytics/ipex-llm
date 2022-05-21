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
from openvino.runtime import Core
from openvino.runtime.passes import Manager
from bigdl.nano.utils.log4Error import invalidInputError


class OpenVINOModel:
    def __init__(self, ie_network: str):
        self.ie_network = None
        self.read_network(ie_network)

    def forward_step(self, *inputs):
        return self.infer_request.infer(list(inputs))

    def read_network(self, model: str):
        core = Core()
        self.ie_network = core.read_model(model=model)
        self.exec_model = core.compile_model(model=self.ie_network, device_name='CPU')
        self.infer_request = self.exec_model.create_infer_request()

    def _save_model(self, path):
        """
        Save PytorchOpenVINOModel to local as xml and bin file

        :param path: Path to save the model.
        """
        path = Path(path)
        invalidInputError(self.ie_network,
                          "self.ie_network shouldn't be None.")
        invalidInputError(path.suffix == ".xml",
                          "Path of openvino model must be with '.xml' suffix.")
        pass_manager = Manager()
        pass_manager.register_pass(pass_name="Serialize",
                                   xml_path=str(path),
                                   bin_path=str(path.with_suffix(".bin")))
        pass_manager.run_passes(self.ie_network)
