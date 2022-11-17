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

from typing import Optional, List, Set, Dict
from importlib.util import find_spec
import subprocess
from .utils import AccelerationOption
from bigdl.nano.utils import CPUInfo


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


def available_acceleration_combination(excludes: Optional[List[str]],
                                       includes: Optional[List[str]],
                                       full_methods: Dict[str, AccelerationOption]):
    '''
    :return: a dictionary states the availablity (if meet depdencies)
    '''
    dependency_checker = {"inc": _inc_checker,
                          "ipex": _ipex_checker,
                          "onnxruntime": _onnxruntime_checker,
                          "openvino": _openvino_checker,
                          "pot": _openvino_checker,
                          "bf16": _bf16_checker}
    if excludes is None:
        exclude_set: Set[str] = set()
    else:
        exclude_set: Set[str] = set(excludes)
        exclude_set.discard("original")

    if includes is None:
        include_set: Set[str] = set(full_methods.keys())
    else:
        include_set: Set[str] = set(includes)
        include_set.add("original")

    available_dict = {}
    for method, option in full_methods.items():
        if method not in include_set:
            continue

        if method in exclude_set:
            continue

        available_iter = True
        for name, value in option.__dict__.items():
            if value is True:
                if name in dependency_checker and not dependency_checker[name]():
                    available_iter = False
        available_dict[method] = available_iter
    return available_dict
