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


from openvino.runtime import Core, serialize
import os


def update_names_of_IR_and_export_blob(model, model_name, dir):
    xml_path = os.path.join(dir, model_name + ".xml")
    model.save(xml_path)
    new_ir_path = os.path.join(dir, model_name + "_new.xml")
    blob_path = os.path.join(dir, model_name + ".blob")

    core = Core()
    core.set_property("NPU", {"NPU_COMPILATION_MODE_PARAMS":
                              "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add"})
    core.set_property("NPU", {"PERFORMANCE_HINT": "LATENCY"})
    model = core.read_model(xml_path)
    inputs = model.inputs
    for idx, input in enumerate(inputs):
        if len(input.names) == 0:
            model.inputs[idx].set_names({f"input_{idx}"})
    outputs = model.outputs
    for idx, input in enumerate(outputs):
        if len(input.names) == 0:
            model.outputs[idx].set_names({f"output_{idx}"})
    # rewrite this model to a new IR path
    if new_ir_path is not None:
        serialize(model, new_ir_path)

    if blob_path is not None:
        compiledModel = core.compile_model(model, device_name="NPU")
        model_stream = compiledModel.export_model()
        with open(blob_path, 'wb') as f:
            f.write(model_stream)

    os.remove(xml_path)
    os.remove(new_ir_path)

    return blob_path
