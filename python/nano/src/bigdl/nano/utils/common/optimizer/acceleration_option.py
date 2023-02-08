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


from abc import abstractmethod
from typing import Optional, List, Set, Dict

from bigdl.nano.utils.common import _inc_checker, _ipex_checker,\
    _onnxruntime_checker, _openvino_checker


_whole_acceleration_options = ["inc", "ipex", "onnxruntime", "openvino", "pot",
                               "bf16", "jit", "channels_last"]


class AccelerationOption(object):
    __slot__ = _whole_acceleration_options

    def __init__(self, **kwargs):
        '''
        initialize optimization option
        '''
        for option in _whole_acceleration_options:
            setattr(self, option, kwargs.get(option, False))
        self.method = kwargs.get("method", None)

    def get_precision(self):
        if self.inc or self.pot:
            return "int8"
        if self.bf16:
            return "bf16"
        return "fp32"

    def get_accelerator(self):
        if self.onnxruntime:
            return "onnxruntime"
        if self.openvino:
            return "openvino"
        if self.jit:
            return "jit"
        return None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass


def available_acceleration_combination(excludes: Optional[List[str]],
                                       includes: Optional[List[str]],
                                       full_methods: Dict[str, AccelerationOption],
                                       all_methods: Dict[str, AccelerationOption] = None):
    '''
    :return: a dictionary states the availablity (if meet depdencies)
    '''
    dependency_checker = {"inc": _inc_checker,
                          "ipex": _ipex_checker,
                          "onnxruntime": _onnxruntime_checker,
                          "openvino": _openvino_checker,
                          "pot": _openvino_checker}
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
        if all_methods is not None:
            for method in include_set:
                if method not in full_methods:
                    # append include method into full methods
                    full_methods[method] = all_methods[method]

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
