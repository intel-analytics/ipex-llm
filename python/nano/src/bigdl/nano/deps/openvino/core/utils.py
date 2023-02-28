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
import subprocess
import operator
from pathlib import Path
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import compare_version
from openvino.runtime.passes import Manager

OpenVINO_LESS_2022_3 = compare_version("openvino", operator.lt, "2022.3")


def convert_onnx_to_xml(onnx_file_path, xml_path, precision,
                        logging=True, batch_size=1, **kwargs):
    xml_path = Path(xml_path)
    model_name, output_dir = str(xml_path.stem), str(xml_path.parent)
    precision_str = "--data_type FP16" if precision == 'fp16' else ""
    params_str = ""
    for key, value in kwargs.items():
        value = str(value)
        value = value.replace(' ', '')  # remove space in param value
        params_str += "--" + str(key) + " " + str(value) + " "
    if OpenVINO_LESS_2022_3:
        logging_str = "--silent" if logging is False else ""
        mo_cmd = "mo -m {0} {1} {2} {3} -n {4} -o {5}".format(
            str(onnx_file_path),
            logging_str,
            precision_str,
            params_str,
            model_name,
            output_dir)
    else:
        mo_cmd = "mo -m {0} --silent {1} {2} {3} -n {4} -o {5}".format(
            str(onnx_file_path),
            not logging,
            precision_str,
            params_str,
            model_name,
            output_dir)

    p = subprocess.Popen(mo_cmd.split())
    p.communicate()
    invalidInputError(not p.returncode,
                      "ModelOptimizer fails to convert {}.".format(str(onnx_file_path)))


def convert_pb_to_xml(pb_file_path, xml_path, precision,
                      logging=True, batch_size=1, **kwargs):
    xml_path = Path(xml_path)
    model_name, output_dir = str(xml_path.stem), str(xml_path.parent)
    precision_str = "--data_type FP16" if precision == 'fp16' else ""
    params_str = ""
    for key, value in kwargs.items():
        value = str(value)
        value = value.replace(' ', '')  # remove space in param value
        params_str += "--" + str(key) + " " + str(value) + " "
    if OpenVINO_LESS_2022_3:
        logging_str = "--silent" if logging is False else ""
        mo_cmd = "mo --saved_model_dir {0} {1} {2} {3} -n {4} -o {5}".format(
            str(pb_file_path),
            logging_str,
            precision_str,
            params_str,
            model_name,
            output_dir)
    else:
        mo_cmd = "mo --saved_model_dir {0} --silent {1} {2} {3} -n {4} -o {5}".format(
            str(pb_file_path),
            not logging,
            precision_str,
            params_str,
            model_name,
            output_dir)

    p = subprocess.Popen(mo_cmd.split())
    p.communicate()
    invalidInputError(not p.returncode,
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
