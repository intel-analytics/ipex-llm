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
from bigdl.nano.pytorch.amp import BF16Model
from bigdl.nano.deps.openvino.openvino_api import PytorchOpenVINOModel, load_openvino_model
from bigdl.nano.deps.ipex.ipex_api import create_IPEXAccelerator, create_IPEXAccelerator_1_9, \
    PytorchIPEXJITModel, PytorchIPEXJITBF16Model, load_ipexjit_model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import PytorchONNXRuntimeModel, \
    load_onnxruntime_model
from bigdl.nano.deps.neural_compressor.inc_api import load_inc_model, quantize as inc_quantize
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

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
        future usage. All model instance is setting to eval mode.

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

        self._direction: str = direction  # save direction as attr
        # record whether calculate accuracy in optimize by this attr
        if validation_data is not None and metric is not None:
            self._calculate_accuracy = True
        else:
            self._calculate_accuracy = False

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

        model.eval()  # change model to eval state

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
                if self._calculate_accuracy:
                    result_map[method]["accuracy"] =\
                        _accuracy_calculate_helper(acce_model,
                                                   metric, validation_data)
                else:
                    result_map[method]["accuracy"] = None

                result_map[method]["model"] = acce_model
            else:
                pass

        self.optimized_model_dict: Dict = result_map
        print("==========================Optimization Results==========================")
        if self._calculate_accuracy:
            for key, value in self.optimized_model_dict.items():
                print("accleration option: {}, latency: {:.4f}ms, accuracy : {:.4f}"
                      .format(key, value["latency"], value["accuracy"]))
        else:
            for key, value in self.optimized_model_dict.items():
                print("accleration option: {}, latency: {:.4f}ms :"
                      .format(key, value["latency"]))

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
                              "There is no optimized model. You should call .optimize() "
                              "before get_best_model()")
        invalidInputError(accelerator in [None, 'onnxruntime', 'openvino', 'jit'],
                          "Only support accelerator 'onnxruntime', 'openvino' and 'jit'.")
        # TODO: include fp16?
        invalidInputError(precision in [None, 'int8', 'bf16'],
                          "Only support precision 'int8', 'bf16'.")
        if accuracy_criterion is not None and not self._calculate_accuracy:
            invalidInputError(False, "If you want to specify accuracy_criterion, you need "
                              "to set metric and validation_data when call 'optimize'.")

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
                if self._direction == "min":
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
                 **export_kwargs):
        """
        Calibrate a Pytorch-Lightning model for post-training quantization.

        :param model:           A model to be quantized. Model type should be an instance of
                                nn.Module.
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', 'bf16', 'fp16', defaults to 'int8'.
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', defaults to None.
                                None means staying in pytorch.
        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.
                                    Required for static quantization.
                                    It's also used as validation dataloader.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param accuracy_criterion:  Tolerable accuracy drop, defaults to None meaning no
                                    accuracy control.
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better': True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, 'higher_is_better':False} means accuracy
                                    must be smaller than 0.99.
        :param approach:    'static' or 'dynamic'.
                            'static': post_training_static_quant,
                            'dynamic': post_training_dynamic_quant.
                            Default: 'static'. OpenVINO supports static mode only.
        :param method:          Method to do quantization. When accelerator=None, supported
            methods: 'fx', 'eager', 'ipex', defaults to 'fx'. If you don't use ipex, suggest using
            'fx' which executes automatic optimizations like fusion. For more information, please
            refer to https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization.
            When accelerator='onnxruntime', supported methods: 'qlinear', 'integer', defaults
            to 'qlinear'. Suggest 'qlinear' for lower accuracy drop if using static quantization.
            More details in https://onnxruntime.ai/docs/performance/quantization.html.
            This argument doesn't take effect for OpenVINO, don't change it for OpenVINO.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, using default config.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param timeout:     Tuning timeout (seconds). Default: None,  which means early stop.
                            Combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: None, which means no tuning.
                            Combine with timeout field to decide when to exit.
                            "timeout=0, max_trials=1" means it will try quantization only once and
                            return satisfying best model.
        :param input_sample:      An input example to convert pytorch model into ONNX/OpenVINO.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. default: True.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return:            A accelerated Pytorch-Lightning Model if quantization is sucessful.
        """
        if precision == 'bf16':
            if accelerator is None:
                if use_ipex:
                    invalidInputError(not TORCH_VERSION_LESS_1_10,
                                      "torch version should >=1.10 to use ipex")
                    use_jit = (accelerator == "jit")
                    channels_last = export_kwargs["channels_last"] \
                        if "channels_last" in export_kwargs else None
                    return PytorchIPEXJITBF16Model(model, input_sample=input_sample,
                                                   use_ipex=use_ipex, use_jit=use_jit,
                                                   channels_last=channels_last)
                bf16_model = BF16Model(model)
                return bf16_model
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid for BF16.".format(accelerator))
        if precision == 'int8':
            if not accelerator or accelerator == 'onnxruntime':
                method_map = {
                    None: {
                        'fx': 'pytorch_fx',
                        'eager': 'pytorch',
                        'ipex': 'pytorch_ipex',
                        None: 'pytorch_fx'  # default
                    },
                    'onnxruntime': {
                        'qlinear': 'onnxrt_qlinearops',
                        'integer': 'onnxrt_integerops',
                        None: 'onnxrt_qlinearops'  # default
                    }
                }
                framework = method_map[accelerator].get(method, None)
                if accelerator == "onnxruntime":
                    if not type(model).__name__ == 'PytorchONNXRuntimeModel':
                        # try to establish onnx model
                        if input_sample is None:
                            # input_sample can be a dataloader
                            input_sample = calib_dataloader
                        model = InferenceOptimizer.trace(
                            model,
                            input_sample=input_sample,
                            accelerator='onnxruntime',
                            **export_kwargs)
                """
                If accelerator==None, quantized model returned should be an object of PytorchModel
                which is defined by neural-compressor containing a `GraphModule` for inference.
                Otherwise accelerator=='onnxruntime', it returns an ONNXModel object. A supported
                model which is able to run on Pytorch or ONNXRuntime can be fetched by
                `quantized_model.model`.
                """
                return inc_quantize(model, calib_dataloader, metric,
                                    framework=framework,
                                    conf=conf,
                                    approach=approach,
                                    tuning_strategy=tuning_strategy,
                                    accuracy_criterion=accuracy_criterion,
                                    timeout=timeout,
                                    max_trials=max_trials,
                                    onnxruntime_session_options=onnxruntime_session_options)

            elif accelerator == 'openvino':
                model_type = type(model).__name__
                if not model_type == 'PytorchOpenVINOModel':
                    if input_sample is None:
                        # input_sample can be a dataloader
                        input_sample = calib_dataloader
                    model = InferenceOptimizer.trace(model,
                                                     input_sample=input_sample,
                                                     accelerator='openvino',
                                                     logging=logging,
                                                     **export_kwargs)
                invalidInputError(type(model).__name__ == 'PytorchOpenVINOModel',
                                  "Invalid model to quantize. Please use a nn.Module or a model "
                                  "from trainer.trance(accelerator=='openvino')")
                drop_type = None
                higher_is_better = None
                maximal_drop = None
                if metric:
                    if not isinstance(accuracy_criterion, dict):
                        accuracy_criterion = {'relative': 0.99, 'higher_is_better': True}

                    drop_type = 'relative' if 'relative' in accuracy_criterion else 'absolute'
                    higher_is_better = accuracy_criterion.get('higher_is_better', None)
                    maximal_drop = accuracy_criterion.get(drop_type, None)

                kwargs = {
                    "metric": metric,
                    "higher_better": higher_is_better,
                    "drop_type": drop_type,
                    "maximal_drop": maximal_drop,
                    "max_iter_num": max_trials,
                    # TODO following two keys are optional, if there is need, we can add them
                    # "n_requests": None,
                    # "sample_size": 300
                }
                return model.pot(calib_dataloader, **kwargs)
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid.".format(accelerator))
        invalidInputError(False,
                          "Precision {} is invalid.".format(precision))

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator: str = None,
              use_ipex: bool = False,
              onnxruntime_session_options=None,
              logging: bool = True,
              **export_kwargs):
        """
        Trace a pytorch model and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param model: An torch.nn.Module model, including pl.LightningModule.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch
                            backend. 'openvino', 'onnxruntime' and 'jit' are supported for now.
        :param use_ipex: whether we use ipex as accelerator for inferencing. default: False.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. default: True.
        :param **kwargs: other extra advanced settings include
                         1. those be passed to torch.onnx.export function, only valid when
                         accelerator='onnxruntime'/'openvino', otherwise will be ignored.
                         2. if channels_last is set and use_ipex=True, we will transform the
                         data to be channels last according to the setting. Defaultly, channels_last
                         will be set to True if use_ipex=True.
        :return: Model with different acceleration.
        """
        invalidInputError(
            isinstance(model, nn.Module) and not isinstance(model, AcceleratedLightningModule),
            "Expect a nn.Module instance that is not traced or quantized"
            "but got type {}".format(type(model))
        )
        if accelerator == 'openvino':  # openvino backend will not care about ipex usage
            return PytorchOpenVINOModel(model, input_sample, logging, **export_kwargs)
        if accelerator == 'onnxruntime':  # onnxruntime backend will not care about ipex usage
            return PytorchONNXRuntimeModel(model, input_sample, onnxruntime_session_options,
                                           **export_kwargs)
        if accelerator == 'jit' or use_ipex:
            if use_ipex:
                invalidInputError(not TORCH_VERSION_LESS_1_10,
                                  "torch version should >=1.10 to use ipex")
            use_jit = (accelerator == "jit")
            channels_last = export_kwargs["channels_last"]\
                if "channels_last" in export_kwargs else None
            return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex,
                                       use_jit=use_jit, channels_last=channels_last)
        invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))


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
    A simple helper to calculate average latency
    '''
    start_time = time.perf_counter()
    time_list = []
    for i in range(iterrun):
        st = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        time_list.append(end - st)
        # at least need 10 iters and try to control calculation
        # time less than 2 min
        if i + 1 >= min(iterrun, 10) and (end - start_time) > 2:
            iterrun = i + 1
            break
    time_list.sort()
    # remove top and least 10% data
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    metric_list = []
    sample_num = 0
    for i, (data_input, target) in enumerate(data):
        metric_list.append(metric(model(data_input), target).numpy() * data_input.shape[0])
        sample_num += data_input.shape[0]
    return np.sum(metric_list) / sample_num


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
