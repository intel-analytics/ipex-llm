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


import numbers
from typing import Dict
import sigfig

from .acceleration_option import AccelerationOption


def format_acceleration_option(method_name: str,
                               full_methods: Dict[str, AccelerationOption]) -> str:
    '''
    Get a string representation for current method's acceleration option
    '''
    option = full_methods[method_name]
    repr_str = ""
    for key, value in option.__dict__.items():
        if value is True:
            if key == "pot" or key == "fx":
                repr_str = repr_str + "int8" + " + "
            else:
                repr_str = repr_str + key + " + "
        elif isinstance(value, str) and value != 'ipex':
            repr_str = repr_str + value + " + "
    if len(repr_str) > 0:
        # remove " + " at last
        repr_str = repr_str[:-3]
    if repr_str == "":
        # if no acceleration is applied, just return "original"
        repr_str = "original"
    return repr_str


def format_optimize_result(optimize_result_dict: dict,
                           calculate_accuracy: bool) -> str:
    '''
    Get a format string representation for optimization result
    '''
    if calculate_accuracy is True:
        horizontal_line = " {0} {1} {2} {3}\n" \
            .format("-" * 32, "-" * 22, "-" * 14, "-" * 22)
        repr_str = horizontal_line
        repr_str += "| {0:^30} | {1:^20} | {2:^12} | {3:^20} |\n" \
            .format("method", "status", "latency(ms)", "metric value")
        repr_str += horizontal_line
        for method, result in optimize_result_dict.items():
            status = result["status"]
            latency = result.get("latency", "None")
            if latency != "None":
                latency = sigfig.round(latency, sigfigs=5)
            accuracy = result.get("accuracy", "None")
            if accuracy != "None" and isinstance(accuracy, float):
                accuracy = sigfig.round(accuracy, sigfigs=5)
            elif isinstance(accuracy, numbers.Real):
                # support more types
                accuracy = float(accuracy)
                accuracy = sigfig.round(accuracy, sigfigs=5)
            else:
                try:
                    import torch
                    # turn Tensor into float
                    if isinstance(accuracy, torch.Tensor):
                        accuracy = accuracy.item()
                        accuracy = sigfig.round(accuracy, sigfigs=5)
                except ImportError:
                    pass
            method_str = f"| {method:^30} | {status:^20} | " \
                         f"{latency:^12} | {accuracy:^20} |\n"
            repr_str += method_str
        repr_str += horizontal_line
    else:
        horizontal_line = " {0} {1} {2}\n" \
            .format("-" * 32, "-" * 22, "-" * 14)
        repr_str = horizontal_line
        repr_str += "| {0:^30} | {1:^20} | {2:^12} |\n" \
            .format("method", "status", "latency(ms)")
        repr_str += horizontal_line
        for method, result in optimize_result_dict.items():
            status = result["status"]
            latency = result.get("latency", "None")
            if latency != "None":
                latency = sigfig.round(latency, sigfigs=5)
            method_str = f"| {method:^30} | {status:^20} | {latency:^12} |\n"
            repr_str += method_str
        repr_str += horizontal_line
    return repr_str
