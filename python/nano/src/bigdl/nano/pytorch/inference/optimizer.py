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

from xml.etree.ElementInclude import include
from torch import nn
import subprocess
from importlib.util import find_spec
import time
import numpy as np

from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch import Trainer

_whole_acceleration_options = ["inc", "ipex", "onnxruntime", "openvino", "pot", 
                               "bf16", "jit", "channels_last"]


class AccelerationOption(object):
    def __init__(self, *args, **kwargs):
        '''
        initialize optimization option
        '''
        for option in _whole_acceleration_options:
            setattr(self, option, kwargs.get(option, False))

    def get_precision(self):
        if self.inc:
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
        "None_fp32": AccelerationOption(),
        "None_fp32_ipex": AccelerationOption(ipex=True),
        "None_bf16": AccelerationOption(bf16=True),
        "None_bf16_ipex": AccelerationOption(bf16=True, ipex=True),
        "None_int8": AccelerationOption(inc=True),
        "jit_fp32": AccelerationOption(jit=True),
        "jit_fp32_ipex": AccelerationOption(jit=True, ipex=True),
        "jit_fp32_ipex_clast": AccelerationOption(jit=True, ipex=True, 
                                                  channels_last=True),
        "jit_bf16": AccelerationOption(jit=True, bf16=True),
        "jit_bf16_clast": AccelerationOption(jit=True, bf16=True,
                                             channels_last=True),
        "jit_bf16_ipex": AccelerationOption(jit=True, bf16=True, ipex=True),
        "jit_bf16_ipex_clast": AccelerationOption(jit=True, bf16=True, 
                                                  ipex=True, channels_last=True),
        "onnxruntime_fp32": AccelerationOption(onnxtunrime=True),
        "onnxruntime_int8_qlinear": AccelerationOption(onnxruntime=True, inc=True),
        "onnxruntime_int8_integer": AccelerationOption(onnxruntime=True, inc=True),
        "openvino_fp32": AccelerationOption(openvino=True),
        "openvino_int8": AccelerationOption(openvino=True, inc=True),
    }


class Optimizer:

    def __init__(self):
        '''
        initialize an optimizer
        '''
        # optimized_model_dict handles the optimized model and some metadata
        # in {"method_name": {"latency": ..., "accuracy": ..., "model": ...}}
        self.optimized_model_dict = {}

    def optimize(self, model,
                 training_data,
                 validation_data=None,
                 metric=None,
                 direction=None,
                 cpu_num=None):
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
               higher the better.
        :param cpu_num: (optional) a int represents how many cores is needed for
               inference.
        '''
        # TODO: direction to be implemented
        # TODO: cpu_num to be implemented

        # check if model is a nn.Module or inherited from a nn.Module
        invalidInputError(isinstance(model, nn.Module), "model should be a nn module.")

        # get the available methods whose dep is met
        available_dict = _available_acceleration_combination()
        result_map = {}

        for method, available in available_dict.items():
            if available:
                instance = ALL_INFERENCE_ACCELERATION_METHOD[method]
                use_ipex = instance.ipex
                accelerator = instance.get_accelerator()
                precision = instance.get_precision()
                # if precision is fp32, then we will use trace method
                if precision == "fp32":
                    input_sample = tuple(next(iter(training_data))[:-1])
                    try:
                        if accelerator is None and use_ipex is False:
                            accelerated_model = model
                        else:
                            # TODO: remove the logging of tracing
                            accelerated_model = Trainer.trace(model=model,
                                                              accelerator=accelerator,
                                                              use_ipex=use_ipex,
                                                              channels_last=instance.channels_last,
                                                              input_sample=input_sample)
                        # TODO: 100 trial run is now fixed, we may make it adjusted intelligently.
                        result_map[method] = {}

                        def func_test(model, input_sample):
                            model(*input_sample)

                        result_map[method]["latency"] =\
                            _throughput_calculate_helper(100, func_test,
                                                         accelerated_model, input_sample)

                        if validation_data is not None and metric is not None:
                            result_map[method]["accuracy"] =\
                                _accuracy_calculate_helper(accelerated_model,
                                                           metric, validation_data)

                        result_map[method]["model"] = accelerated_model

                    except Exception as e:
                        print(e)

                # if precision is int8 or bf16, then we will use quantize method
                if precision in ("int8", "bf16"):
                    ort_method = _detect_ort_method(method)
                    try:
                        # TODO: remove the logging of quantization
                        accelerated_model = Trainer.quantize(model=model,
                                                             precision=precision,
                                                             accelerator=accelerator,
                                                             use_ipex=use_ipex,
                                                             calib_dataloader=training_data,
                                                             method=ort_method)
                        result_map[method] = {}

                        def func_test(model, input_sample):
                            model(*input_sample)

                        result_map[method]["latency"] =\
                            _throughput_calculate_helper(100, func_test,
                                                         accelerated_model, input_sample)

                        if validation_data is not None and metric is not None:
                            result_map[method]["accuracy"] =\
                                _accuracy_calculate_helper(accelerated_model,
                                                           metric, validation_data)

                        result_map[method]["model"] = accelerated_model

                    except Exception as e:
                        print(e)
            else:
                pass

        self.optimized_model_dict = result_map

    def get_best_model(self,
                       accelerator=None,
                       precision=None,
                       use_ipex=None,
                       allow_acc=None,):
        '''
        :param accelerator: (optional) if not None, then will only find the
               model with this specific accelerator.
        :param precision: (optional) if not None, the will only find the
               model with thie specific precision.
        :param use_ipex: (optional) if not NOne, then will only find the
               model with this specific ipex setting
        :param allow_acc: (optional) a float represents the accuracy threshold
               that can be tollerated.

        :return: best model
        '''
        pass


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


def _detect_ort_method(method_name):
    method_name = method_name.split("_") [-1]
    if method_name in ["qlinear", "integer"]:
        return method_name
    return None
    

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
            if value:
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
    # TODO: remove first and last 10 percent
    return np.median(time_list) * 1000


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    metric_list = []
    # TODO: data should have same batchsize
    for i, (data_input, target) in enumerate(data):
        metric_list.append(metric(model(data_input), target).numpy())
    return np.mean(metric_list)
