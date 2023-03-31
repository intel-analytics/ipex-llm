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

from importlib.util import find_spec
from bigdl.nano.utils.common import CPUInfo
import os
import subprocess
from bigdl.nano.utils.common import invalidInputError


def _inc_checker():
    '''
    check if intel neural compressor is installed
    '''
    return not find_spec("neural_compressor") is None


def _ipex_checker():
    '''
    check if intel pytorch extension is installed
    '''
    return not find_spec("intel_extension_for_pytorch") is None


def _onnxruntime_checker():
    '''
    check if onnxruntime and onnx is installed
    '''
    onnxruntime_installed = not find_spec("onnxruntime") is None
    onnx_installed = not find_spec("onnx") is None
    return onnxruntime_installed and onnx_installed


def _openvino_checker():
    '''
    check if openvino-dev is installed
    '''
    return not find_spec("openvino") is None


def _bf16_checker():
    '''
    bf16 availablity will be decided dynamically during the optimization
    '''
    cpuinfo = CPUInfo()
    return cpuinfo.has_bf16


def _avx512_checker():
    '''
    check if avx512 instruction is supported
    '''
    cpuinfo = CPUInfo()
    return cpuinfo.has_avx512


def get_cpu_info():
    # copied from https://github.com/intel/intel-extension-for-pytorch
    cpu_info_path = "/proc/cpuinfo"
    if os.path.exists(cpu_info_path):
        cpu_info_command = "cat {}".format(cpu_info_path)
        all_sub_cpu_info = subprocess.getoutput(cpu_info_command).strip()
        for sub_cpu_info in all_sub_cpu_info.split("\n"):
            if sub_cpu_info.startswith("flags"):
                cpu_flags = sub_cpu_info.replace("\t", '').upper().split(":")
                invalidInputError(len(cpu_flags) >= 2, "Less than two CPU flags.")
                all_cpu_flags = cpu_flags[1].split(" ")
                return all_cpu_flags

        return []
    else:
        sys.exit("The extension does not support current platform - {}.".format(platform.system()))


def _avx2_checker():
    # copied from https://github.com/intel/intel-extension-for-pytorch
    cpu_flags = get_cpu_info()
    minimal_binary_isa = "AVX2"
    if minimal_binary_isa not in cpu_flags:
        return False

    return True
