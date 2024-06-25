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

import tempfile
from tempfile import TemporaryDirectory
from pathlib import Path
import torch
import openvino as ov
from openvino.runtime import Core


def get_core():
    core = Core()
    core.set_property({'CACHE_DIR': "blob_npu"})
    core.set_property("NPU", {
        "NPU_COMPILER_TYPE": "DRIVER",
        "PERFORMANCE_HINT": "LATENCY"})
    return core

OV_CORE = get_core()

compiler_args = [
    "vertical-fusion=true",
    "dpu-profiling=false",
    "dma-profiling=false",
    "sw-profiling=false",
    "dump-task-stats=true",
    "enable-schedule-trace=false",
    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add"
]
NPU_CONFIG = {
    "NPU_COMPILER_TYPE": "DRIVER",
    "NPU_COMPILATION_MODE": "DefaultHW",
    "PERF_COUNT": "NO",
    "NPU_COMPILATION_MODE_PARAMS": " ".join(compiler_args),
    "NPU_USE_ELF_COMPILER_BACKEND": "YES",
    "PERFORMANCE_HINT": "LATENCY"}


def get_npu_model(model, inputs, input_names, output_names, save_dir=None, quantize=False, device="NPU"):

    # create a temp directory to save the model
    with tempfile.TemporaryDirectory() as tmpdirname:
        if save_dir is None:
            save_dir = tmpdirname
        save_dir_path = Path(save_dir)
        # todo try to remove onnx export
        torch.onnx.export(model, inputs, save_dir_path / 'model.onnx',
                          input_names=input_names, output_names=output_names)
        ov_model = ov.convert_model(save_dir_path / 'model.onnx')
        if quantize:
            from nncf.parameters import ModelType
            import nncf
            dataset = nncf.Dataset(torch.utils.data.DataLoader(inputs, batch_size=1))
            ov_model = nncf.quantize(ov_model, dataset, model_type=ModelType.TRANSFORMER)
        if device == "NPU":
            compiled_model = OV_CORE.compile_model(ov_model, "NPU", config=NPU_CONFIG)
        elif device == "GPU":
            compiled_model = OV_CORE.compile_model(ov_model, "GPU")
        else:
            compiled_model = OV_CORE.compile_model(ov_model, "CPU")
    # ov.save_model(ov_model, save_dir_path / 'model.xml')
    # compiled_model = OV_CORE.compile_model(ov_model, "CPU")
    return compiled_model, OV_CORE
