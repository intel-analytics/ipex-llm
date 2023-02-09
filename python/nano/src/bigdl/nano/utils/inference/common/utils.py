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
import sys
import tempfile
from collections import namedtuple
import os
import time
import numbers

import cloudpickle
import numpy as np
from typing import Dict, Callable, Optional
from abc import abstractmethod
import bigdl
from . import _worker

_whole_acceleration_options = ["inc", "ipex", "onnxruntime", "openvino", "pot",
                               "bf16", "jit", "channels_last"]

_whole_acceleration_env = ['tcmalloc', 'jemalloc', 'openmp', 'perf']

CompareMetric = namedtuple("CompareMetric", ["method_name", "latency", "accuracy"])


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


class AccelerationEnv(object):
    __slot__ = _whole_acceleration_env
    _CONDA_DIR = os.environ.get('CONDA_PREFIX', None)
    _NANO_DIR = os.path.join(os.path.dirname(bigdl.__file__), 'nano')

    def __init__(self, **kwargs):
        """
        initialize optimization env
        """
        self.openmp = None
        self.jemalloc = None
        self.tcmalloc = None
        self.perf = None
        for option in _whole_acceleration_env:
            setattr(self, option, kwargs.get(option, False))

    def get_malloc_lib(self):
        if self.tcmalloc:
            return "tcmalloc"
        if self.jemalloc:
            return "jemalloc"
        return None

    def get_omp_lib(self):
        if self.openmp and self.perf:
            return "openmp_perf"
        elif self.openmp:
            return "openmp"
        else:
            return None

    def get_env_dict(self):
        tmp_env_dict = {}

        # set allocator env var
        tmp_malloc_lib = self.get_malloc_lib()
        if tmp_malloc_lib == 'jemalloc':
            tmp_env_dict['LD_PRELOAD'] = os.path.join(
                AccelerationEnv._NANO_DIR, 'libs/libjemalloc.so')
            tmp_env_dict['MALLOC_CONF'] = 'oversize_threshold:1,background_thread:false,' \
                                          'metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1'
        elif tmp_malloc_lib:
            tmp_env_dict['LD_PRELOAD'] = os.path.join(
                AccelerationEnv._NANO_DIR, 'libs/libtcmalloc.so')
            tmp_env_dict['MALLOC_CONF'] = ''
        else:
            tmp_env_dict['LD_PRELOAD'] = ''
            tmp_env_dict['MALLOC_CONF'] = ''

        # set omp env var
        omp_lib_path = ''
        if AccelerationEnv._CONDA_DIR:
            if os.path.exists(os.path.join(AccelerationEnv._CONDA_DIR, '../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._CONDA_DIR, '../lib/libiomp5.so')
            elif os.path.exists(os.path.join(AccelerationEnv._CONDA_DIR,
                                             '../../../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._CONDA_DIR,
                                            '../../../lib/libiomp5.so')
        elif AccelerationEnv._NANO_DIR:
            if os.path.exists(os.path.join(AccelerationEnv._NANO_DIR,
                                           '../../../libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._NANO_DIR,
                                            '../../../libiomp5.so')
            elif os.path.exists(os.path.join(AccelerationEnv._NANO_DIR,
                                             '../../../../../../lib/libiomp5.so')):
                omp_lib_path = os.path.join(AccelerationEnv._NANO_DIR,
                                            '../../../../../../lib/libiomp5.so')

        tmp_omp_lib = self.get_omp_lib()
        if tmp_omp_lib == 'openmp_perf':
            tmp_env_dict['LD_PRELOAD'] = tmp_env_dict['LD_PRELOAD'] + ' ' + omp_lib_path
            tmp_env_dict['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
            tmp_env_dict['KMP_BLOCKTIME'] = '1'
        elif tmp_omp_lib == 'openmp':
            tmp_env_dict['LD_PRELOAD'] = tmp_env_dict['LD_PRELOAD'] + ' ' + omp_lib_path
            tmp_env_dict['KMP_AFFINITY'] = 'granularity=fine,none'
            tmp_env_dict['KMP_BLOCKTIME'] = '1'
        else:
            tmp_env_dict['KMP_AFFINITY'] = ''
            tmp_env_dict['KMP_BLOCKTIME'] = ''
        return tmp_env_dict


def throughput_calculate_helper(iterrun, baseline_time, func, *args):
    '''
    A simple helper to calculate average latency
    '''
    # test run two times for more accurate latency
    for i in range(2):
        func(*args)
    start_time = time.perf_counter()
    time_list = []
    for i in range(iterrun):
        st = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        time_list.append(end - st)
        # if three samples cost more than 4x time than baseline model, prune it
        if i == 2 and end - start_time > 12 * baseline_time:
            return np.mean(time_list) * 1000, False
        # at least need 10 iters and try to control calculation
        # time less than 10s
        if i + 1 >= min(iterrun, 10) and (end - start_time) > 10:
            iterrun = i + 1
            break
    time_list.sort()
    # remove top and least 10% data
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000, True


def format_acceleration_option(method_name: str,
                               full_methods: Dict[str, AccelerationOption]) -> str:
    '''
    Get a string represation for current method's acceleration option
    '''
    option = full_methods[method_name]
    repr_str = ""
    for key, value in option.__dict__.items():
        if value is True:
            if key == "pot":
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
    Get a format string represation for optimization result
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
                latency = round(latency, 3)
            accuracy = result.get("accuracy", "None")
            if accuracy != "None" and isinstance(accuracy, float):
                accuracy = round(accuracy, 3)
            elif isinstance(accuracy, numbers.Real):
                # support more types
                accuracy = float(accuracy)
                accuracy = round(accuracy, 3)
            else:
                try:
                    import torch
                    # turn Tensor into float
                    if isinstance(accuracy, torch.Tensor):
                        accuracy = accuracy.item()
                        accuracy = round(accuracy, 3)
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
                latency = round(latency, 3)
            method_str = f"| {method:^30} | {status:^20} | {latency:^12} |\n"
            repr_str += method_str
        repr_str += horizontal_line
    return repr_str


def exec_with_worker(func: Callable, *args, env: Optional[dict] = None):
    """
    Call func on subprocess with provided environment variables.

    :param func: a Callable object
    :param args: arguments for the func call
    :param env: a mapping that defines the environment variables for the subprocess
    """
    worker_path = _worker.__file__
    tmp_env = {}
    tmp_env.update(os.environ)
    if env is not None:
        tmp_env.update(env)
    if 'PYTHONPATH' in tmp_env:
        tmp_env['PYTHONPATH'] = ":".join([tmp_env['PYTHONPATH'], *sys.path])
    else:
        tmp_env['PYTHONPATH'] = ":".join(sys.path)
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        param_file = os.path.join(tmp_dir_path, 'param')
        with open(param_file, 'wb') as f:
            cloudpickle.dump([func, *args], f)
        subprocess.run(["python", worker_path, param_file],
                       check=True, env=tmp_env)
        with open(os.path.join(tmp_dir_path, _worker.RETURN_FILENAME), 'rb') as f:
            return cloudpickle.load(f)
