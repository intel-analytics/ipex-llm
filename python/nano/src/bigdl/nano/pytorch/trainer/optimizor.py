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

from torch import nn
from bigdl.nano.utils.log4Error import invalidInputError
from importlib.util import find_spec
import time
import numpy as np
from bigdl.nano.pytorch import Trainer


ALL_INFERENCE_ACCELERATION_METHOD = \
    {
        "None_fp32_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                             "openvino":False, "pot": False, "bf16": False},
        "None_fp32_ipex": {"inc": False, "ipex": True, "onnxruntime": False,
                           "openvino":False, "pot": False, "bf16": False},
        "None_bf16_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                             "openvino":False, "pot": False, "bf16": True},
        "None_bf16_ipex": {"inc": False, "ipex": True, "onnxruntime": False,
                           "openvino":False, "pot": False, "bf16": True},
        "None_int8_noipex": {"inc": True, "ipex": False, "onnxruntime": False,
                             "openvino":False, "pot": False, "bf16": False},
        "jit_fp32_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                            "openvino":False, "pot": False, "bf16": False},
        "jit_fp32_ipex": {"inc": False, "ipex": True, "onnxruntime": False,
                          "openvino":False, "pot": False, "bf16": False},
        "jit_bf16_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                            "openvino":False, "pot": False, "bf16": True},
        "jit_bf16_ipex": {"inc": False, "ipex": True, "onnxruntime": False,
                          "openvino":False, "pot": False, "bf16": True},
        "onnxruntime_fp32_noipex": {"inc": False, "ipex": False, "onnxruntime": True,
                                    "openvino":False, "pot": False, "bf16": False},
        "onnxruntime_int8_noipex_qlinear": {"inc": True, "ipex": False, "onnxruntime": True,
                                            "openvino":False, "pot": False, "bf16": False},
        "onnxruntime_int8_noipex_integer": {"inc": True, "ipex": False, "onnxruntime": True,
                                            "openvino":False, "pot": False, "bf16": False},
        "openvino_fp32_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                                 "openvino":True, "pot": False, "bf16": False},
        "openvino_int8_noipex": {"inc": False, "ipex": False, "onnxruntime": False,
                                 "openvino":True, "pot": True, "bf16": False},
    }


class Optimizor:

    def __init__(self):
        '''
        initialize an optimizor
        '''
        self.optimized_model_dict = {}

    def optimize(model,
                 training_data,
                 validation_data=None,
                 metric=None,
                 direction=None,
                 cpu_num=None):
        '''
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
               better for the metric
        :param cpu_num: (optional) a int represents how many cores is needed for
               inference.
        '''
        # TODO: direction to be implemented
        # TODO: cpu_num to be implemented
        invalidInputError(isinstance(model, nn.Module), "model should be a nn module.")

        available_dict = _available_acceleration_combination()
        result_map = {}
        
        for method, available in available_dict.items():
            if available:
                split_method = method.split('_')
                use_ipex = (split_method[2] == "ipex")
                accelerator = None if split_method[0] == "None" else split_method[0]
                if split_method[1] == "fp32":  # trace
                    input_sample = tuple(next(iter(training_data))[:-1])
                    try:
                        if accelerator is None and use_ipex is False:
                            accelerated_model = model
                        else:
                            accelerated_model = Trainer.trace(model=model,
                                                            accelerator=accelerator,
                                                            use_ipex=use_ipex,
                                                            input_sample=input_sample)
                        def func_test(model, input_sample):
                            model(*input_sample)
                        # TODO: 100 trial run is now fixed, we may make it adjusted intelligently.
                        result_map[method] = {"latency": _throughput_calculate_helper(100, func_test, accelerated_model, input_sample)}
                        if validation_data is not None and metric is not None:
                            result_map[method]["accuracy"] = _accuracy_calculate_helper(accelerated_model, metric, validation_data)
                        result_map[method]["model"] = accelerated_model
                    except Exception as e:
                        print(e)
                if split_method[1] == "int8" or split_method[1] == "bf16":  # quantize
                    precision = split_method[1]
                    q_method = None if len(split_method) < 4 else split_method[3]
                    try:
                        accelerated_model = Trainer.quantize(model=model,
                                                             precision=precision,
                                                             accelerator=accelerator,
                                                             use_ipex=use_ipex,
                                                             calib_dataloader=training_data,
                                                             method=q_method)
                        # TODO: exclude bf16 model if isa is not supported to avoid useless searching
                        def func_test(model, input_sample):
                            model(*input_sample)
                        result_map[method] = {"latency": _throughput_calculate_helper(100, func_test, accelerated_model, input_sample)}
                        if validation_data is not None and metric is not None:
                            result_map[method]["accuracy"] = _accuracy_calculate_helper(accelerated_model, metric, validation_data)
                        result_map[method]["model"] = accelerated_model
                    except Exception as e:
                        print(e)
            else:
                pass
        for iter_item in sorted(result_map.items(), key=lambda x:x[1]["latency"]):
            print(iter_item)
        self.optimized_model_dict = result_map

    def export(accelerator=None,
               precision=None,
               use_ipex=None,
               allow_acc=None,):
        '''
        :param accelerator: (optional) if not None, then will only find the
               model with this specific accelerator.
        :param precision: (optional) if not None, the will only find the
               model with thie specific precision.
        :param use_ipex: (optional) if not None, then will only find the
               model with this specific ipex setting
        :param allow_acc: (optional) a float represents the accuracy threshold
               that can be tollerated.
        '''
        pass

    def get_best_model(accelerator=None,
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
    return True


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
    available_dict={}
    for method, dep in ALL_INFERENCE_ACCELERATION_METHOD.items():
        available_iter = True
        for name, value in dep.items():
            if value:
                if not dependency_checker[name]():
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
