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

from collections import namedtuple
import torch
from torch import nn
import subprocess
from importlib.util import find_spec
import time
import numpy as np
from copy import deepcopy
from typing import Dict, Callable, Tuple
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from bigdl.nano.utils.log4Error import invalidInputError, invalidOperationError
from bigdl.nano.pytorch import Trainer
import os
os.environ['LOGLEVEL'] = 'ERROR'  # remove parital output of inc


_whole_acceleration_options = ["inc", "ipex", "onnxruntime", "openvino", "pot",
                               "bf16", "jit", "channels_last"]

CompareMetric = namedtuple("CompareMetric", ["method_name", "latency", "accuracy"])


class AccelerationOption(object):
    __slot__ = _whole_acceleration_options

    def __init__(self, *args, **kwargs):
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


# acceleration method combinations, developers may want to register some new
# combinations here
ALL_INFERENCE_ACCELERATION_METHOD = \
    {
        "original": AccelerationOption(),
        "fp32_ipex": AccelerationOption(ipex=True),
        "bf16": AccelerationOption(bf16=True),
        "bf16_ipex": AccelerationOption(bf16=True, ipex=True),
        "int8": AccelerationOption(inc=True),
        "jit_fp32": AccelerationOption(jit=True),
        "jit_fp32_ipex": AccelerationOption(jit=True, ipex=True),
        "jit_fp32_ipex_clast": AccelerationOption(jit=True, ipex=True,
                                                  channels_last=True),
        "openvino_fp32": AccelerationOption(openvino=True),
        "openvino_int8": AccelerationOption(openvino=True, pot=True),
        "onnxruntime_fp32": AccelerationOption(onnxtunrime=True),
        "onnxruntime_int8_qlinear": AccelerationOption(onnxruntime=True, inc=True,
                                                       method="qlinear"),
        "onnxruntime_int8_integer": AccelerationOption(onnxruntime=True, inc=True,
                                                       method="integer"),
    }


class InferenceOptimizer:

    def __init__(self):
        '''
        initialize an optimizer
        '''
        # optimized_model_dict handles the optimized model and some metadata
        # in {"method_name": {"latency": ..., "accuracy": ..., "model": ...}}
        self.optimized_model_dict = {}

    def optimize(self, model: nn.Module,
                 training_data: DataLoader,
                 validation_data: DataLoader = None,
                 metric: Callable = None,
                 direction: str = "max",
                 cpu_num: int = None,
                 logging: bool = False,
                 latency_sample_num: int = 100) -> None:
        '''
        This function will give all available inference acceleration methods a try
        and record the latency, accuracy and model instance inside the Optimizer for
        future usage.

        :param model: A nn.module to be optimized
        :param training_data: A pytorch dataloader for training dataset.
               Users should be careful with this parameter since this dataloader
               might be exposed to the model, which causing data leak. The
               batch_size of this dataloader is important as well, users may
               want to set it to the same batch size you may want to use the model
               in real deploy environment. E.g. batch size should be set to 1
               if you would like to use the accelerated model in an online service.
        :param validation_data: (optional) A pytorch dataloader for accuracy evaluation
               This is only needed when users care about the possible accuracy drop.
        :param metric: (optional) A callable object takes prediction and target
               and returns a accuracy value in this calling method `metric(pred, target)`
        :param direction: (optional) A string that indicates the higher/lower
               better for the metric, "min" for the lower the better and "max" for the
               higher the better. Default value is "max".
        :param cpu_num: (optional) a int represents how many cores is needed for
               inference.
        :param logging: whether to log detailed information of model conversion.
               default: False.
        :param latency_sample_num: (optional) a int represents the number of repetitions
               to calculate the average latency. The default value is 100.
        '''
        # TODO: may support accuracy_criterion

        # check if model is a nn.Module or inherited from a nn.Module
        invalidInputError(isinstance(model, nn.Module), "model should be a nn module.")
        invalidInputError(direction in ['min', 'max'],
                          "Only support direction 'min', 'max'.")

        # get the available methods whose dep is met
        available_dict: Dict = _available_acceleration_combination()

        self.direction: str = direction  # save direction as attr

        default_threads: int = torch.get_num_threads()
        cpu_num: int = default_threads if cpu_num is None else int(cpu_num)

        # set cpu num for onnxruntime
        if _onnxruntime_checker():
            import onnxruntime
            sessoption = onnxruntime.SessionOptions()
            sessoption.intra_op_num_threads = cpu_num
            sessoption.inter_op_num_threads = cpu_num
        else:
            sessoption = None
        # TODO: set cpu num for openvino

        result_map: Dict[str, Dict] = {}

        for method, available in available_dict.items():
            if available:
                option: AccelerationOption = ALL_INFERENCE_ACCELERATION_METHOD[method]
                use_ipex: bool = option.ipex
                use_channels_last: bool = option.channels_last
                accelerator: str = option.get_accelerator()
                precision: str = option.get_precision()
                # if precision is fp32, then we will use trace method
                if precision == "fp32":
                    input_sample = tuple(next(iter(training_data))[:-1])
                    try:
                        if accelerator is None and use_ipex is False:
                            acce_model = model
                        else:
                            if accelerator in ("jit", None):
                                acce_model = \
                                    InferenceOptimizer.trace(model=model,
                                                             accelerator=accelerator,
                                                             use_ipex=use_ipex,
                                                             # channels_last is only for jit
                                                             channels_last=use_channels_last,
                                                             input_sample=input_sample)
                            else:
                                acce_model = \
                                    InferenceOptimizer.trace(model=model,
                                                             accelerator=accelerator,
                                                             input_sample=input_sample,
                                                             onnxruntime_session_options=sessoption,
                                                             # remove output of openvino
                                                             logging=logging)
                    except Exception as e:
                        print(e)
                        continue

                # if precision is int8 or bf16, then we will use quantize method
                elif precision in ("int8", "bf16"):
                    ort_method: str = option.method
                    try:
                        acce_model = \
                            InferenceOptimizer.quantize(model=deepcopy(model),
                                                        precision=precision,
                                                        accelerator=accelerator,
                                                        use_ipex=use_ipex,
                                                        calib_dataloader=training_data,
                                                        method=ort_method,
                                                        onnxruntime_session_options=sessoption,
                                                        # remove output of openvino
                                                        logging=logging)
                    except Exception as e:
                        print(e)
                        continue

                result_map[method] = {}

                def func_test(model, input_sample):
                    model(*input_sample)

                torch.set_num_threads(cpu_num)
                try:
                    result_map[method]["latency"] =\
                        _throughput_calculate_helper(latency_sample_num, func_test,
                                                     acce_model, input_sample)
                except Exception as e:
                    result_map.pop(method)
                    torch.set_num_threads(default_threads)
                    continue

                torch.set_num_threads(default_threads)
                if validation_data is not None and metric is not None:
                    result_map[method]["accuracy"] =\
                        _accuracy_calculate_helper(acce_model,
                                                   metric, validation_data)

                result_map[method]["model"] = acce_model

            else:
                pass

        self.optimized_model_dict: Dict = result_map

    def get_best_model(self,
                       accelerator: str = None,
                       precision: str = None,
                       use_ipex: bool = None,
                       accuracy_criterion: float = None) -> Tuple[nn.Module, str]:
        '''
        :param accelerator: (optional) Use accelerator 'None', 'onnxruntime',
               'openvino', 'jit', defaults to None. If not None, then will only find the
               model with this specific accelerator.
        :param precision: (optional) Supported type: 'int8', 'bf16',
               defaults to None which represents 'fp32'. If not None, the will
               only find the model with thie specific precision.
        :param use_ipex: (optional) if not NOne, then will only find the
               model with this specific ipex setting
        :param :param accuracy_criterion: (optional) a float represents tolerable
               accuracy drop percentage, defaults to None meaning no accuracy control.
        :return: best model, corresponding acceleration option
        '''
        invalidOperationError(len(self.optimized_model_dict) > 0,
                              "There is no optimized model. You should call .optimize() \
                              before get_best_model()")
        invalidInputError(accelerator in [None, 'onnxruntime', 'openvino', 'jit'],
                          "Only support accelerator 'onnxruntime', 'openvino' and 'jit'.")
        # TODO: include fp16?
        invalidInputError(precision in [None, 'int8', 'bf16'],
                          "Only support precision 'int8', 'bf16'.")

        best_model = self.optimized_model_dict["original"]["model"]
        best_metric = CompareMetric("original",
                                    self.optimized_model_dict["original"]["latency"],
                                    self.optimized_model_dict["original"]["accuracy"])

        for method in self.optimized_model_dict.keys():
            if method == "original":
                continue
            option: AccelerationOption = ALL_INFERENCE_ACCELERATION_METHOD[method]
            result: Dict = self.optimized_model_dict[method]
            if accelerator is not None:
                if not getattr(option, accelerator):
                    continue
            if precision is not None:
                if precision == 'bf16' and not option.bf16:
                    continue
                if precision == 'int8' and not (option.inc or option.pot):
                    continue
            if use_ipex:
                if not option.ipex:
                    continue

            if accuracy_criterion is not None:
                accuracy: float = result["accuracy"]
                compare_acc: float = best_metric.accuracy
                if self.direction == "min":
                    if (accuracy - compare_acc) / compare_acc > accuracy_criterion:
                        continue
                else:
                    if (compare_acc - accuracy) / compare_acc > accuracy_criterion:
                        continue

            # After the above conditions are met, the latency comparison is performed
            if result["latency"] < best_metric.latency:
                best_model = result["model"]
                best_metric = CompareMetric(method, result["latency"], result["accuracy"])

        return best_model, _format_acceleration_info(best_metric.method_name)

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator: str = None,
              use_ipex: bool = False,
              onnxruntime_session_options=None,
              logging: bool = True,
              **export_kwargs) -> nn.Module:
        return Trainer.trace(model=model,
                             input_sample=input_sample,
                             accelerator=accelerator,
                             use_ipex=use_ipex,
                             onnxruntime_session_options=onnxruntime_session_options,
                             logging=logging,
                             **export_kwargs)

    @staticmethod
    def quantize(model: nn.Module,
                 precision: str = 'int8',
                 accelerator: str = None,
                 use_ipex: bool = False,
                 calib_dataloader: DataLoader = None,
                 metric: Metric = None,
                 accuracy_criterion: dict = None,
                 approach: str = 'static',
                 method: str = None,
                 conf: str = None,
                 tuning_strategy: str = None,
                 timeout: int = None,
                 max_trials: int = None,
                 input_sample=None,
                 onnxruntime_session_options=None,
                 logging: bool = True,
                 **export_kwargs) -> nn.Module:
        return Trainer.quantize(model=model,
                                precision=precision,
                                accelerator=accelerator,
                                use_ipex=use_ipex,
                                calib_dataloader=calib_dataloader,
                                metric=metric,
                                accuracy_criterion=accuracy_criterion,
                                approach=approach,
                                method=method,
                                conf=conf,
                                tuning_strategy=tuning_strategy,
                                timeout=timeout,
                                max_trials=max_trials,
                                input_sample=input_sample,
                                onnxruntime_session_options=onnxruntime_session_options,
                                logging=logging,
                                **export_kwargs)


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
    return not find_spec("openvino-dev") is None


def _bf16_checker():
    '''
    bf16 availablity will be decided dynamically during the optimization
    '''
    msg = subprocess.check_output(["lscpu"]).decode("utf-8")
    return "avx512_bf16" in msg or "amx_bf16" in msg


def _available_acceleration_combination():
    '''
    :return: a dictionary states the availablity (if meet depdencies)
    '''
    dependency_checker = {"inc": _inc_checker,
                          "ipex": _ipex_checker,
                          "onnxruntime": _onnxruntime_checker,
                          "openvino": _openvino_checker,
                          "pot": _openvino_checker,
                          "bf16": _bf16_checker}
    available_dict = {}
    for method, option in ALL_INFERENCE_ACCELERATION_METHOD.items():
        available_iter = True
        for name, value in option.__dict__.items():
            if value is True:
                if name in dependency_checker and not dependency_checker[name]():
                    available_iter = False
        available_dict[method] = available_iter
    return available_dict


def _throughput_calculate_helper(iterrun, func, *args):
    '''
    A simple helper to calculate median latency
    '''
    time_list = []
    for _ in range(iterrun):
        st = time.time()
        func(*args)
        time_list.append(time.time() - st)
    time_list.sort()
    # remove top and least 10% data.ni
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    metric_list = []
    # TODO: data should have same batchsize
    for i, (data_input, target) in enumerate(data):
        metric_list.append(metric(model(data_input), target).numpy())
    return np.mean(metric_list)


def _format_acceleration_info(method_name):
    '''
    Get a string represation for current method's acceleration option
    '''
    option = ALL_INFERENCE_ACCELERATION_METHOD[method_name]
    repr_str = ""
    for key, value in option.__dict__.items():
        if value is True:
            repr_str = repr_str + key + " + "
        elif isinstance(value, str):
            repr_str = repr_str + value + " + "
    if len(repr_str) > 0:
        repr_str = repr_str[:-2]
    return repr_str
