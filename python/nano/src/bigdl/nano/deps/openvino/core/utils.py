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
from pathlib import Path
from bigdl.nano.utils.log4Error import invalidInputError
from openvino.runtime.passes import Manager


def convert_onnx_to_xml(onnx_file_path, xml_path, batch_size=1):
    xml_path = Path(xml_path)
    model_name, output_dir = str(xml_path.stem), str(xml_path.parent)
    mo_cmd = "mo -m {} -n {} -o {}".format(str(onnx_file_path), model_name, output_dir)
    if os.system(mo_cmd) == 0:
        return
    else:
        invalidInputError(False,
                          "ModelOptimizer fails to convert {}.".format(str(onnx_file_path)))


def convert_pb_to_xml(pb_file_path, xml_path, batch_size=1):
    xml_path = Path(xml_path)
    model_name, output_dir = str(xml_path.stem), str(xml_path.parent)
    mo_cmd = "mo --saved_model_dir {} -n {} -o {}".format(str(pb_file_path), model_name, output_dir)
    if os.system(mo_cmd) == 0:
        return
    else:
        invalidInputError(False,
                          "ModelOptimizer fails to convert {}.".format(str(pb_file_path)))


def save(model, xml_path):
    xml_path = Path(xml_path)
    pass_manager = Manager()
    pass_manager.register_pass(pass_name="Serialize",
                               xml_path=str(xml_path),
                               bin_path=str(xml_path.with_suffix(".bin")))
    pass_manager.run_passes(model)


def validate_dataloader(model, dataloader):
    n_inputs = len(model.ie_network.inputs)
    sample = dataloader[0][:n_inputs]
    try:
        model.forward_step(*sample)
    except RuntimeError:
        invalidInputError(False,
                          "Invalid dataloader, please check if the model inputs are compatible"
                          "with the model.")
