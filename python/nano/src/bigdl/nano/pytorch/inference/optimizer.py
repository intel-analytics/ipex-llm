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

import torch
from torch import nn
import time
from copy import deepcopy
import multiprocessing as mp
from typing import Dict, Callable, Tuple, Optional, List, Union, Sequence
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from bigdl.nano.utils.inference.common.checker import available_acceleration_combination
from bigdl.nano.utils.inference.common.utils import AccelerationOption,\
    throughput_calculate_helper, format_optimize_result
from bigdl.nano.utils.inference.common.base_optimizer import BaseInferenceOptimizer
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.amp import BF16Model
from bigdl.nano.deps.openvino.openvino_api import PytorchOpenVINOModel
from bigdl.nano.deps.ipex.ipex_api import PytorchIPEXJITModel, PytorchIPEXJITBF16Model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import PytorchONNXRuntimeModel
from bigdl.nano.deps.neural_compressor.inc_api import quantize as inc_quantize
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.inference.pytorch.model_utils import get_forward_args, get_input_example
from bigdl.nano.utils.inference.pytorch.metrics import NanoMetric
from bigdl.nano.utils.inference.pytorch.dataset import RepeatDataset, remove_batch_dim_fn
from bigdl.nano.utils.inference.pytorch.dataloader import\
    transform_multiple_input_dataloader_to_inc_mode
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, save_model, load_model
from bigdl.nano.common.cpu_schedule import schedule_processors

from .multi_instance import _MultiInstanceModel, _multi_instance_helper

import traceback
import warnings
# Filter out useless Userwarnings
warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')

import os
os.environ['LOGLEVEL'] = 'ERROR'  # remove parital output of inc


class TorchAccelerationOption(AccelerationOption):
    def optimize(self, model, training_data=None, input_sample=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        accelerator = self.get_accelerator()
        if self.get_precision() == "fp32":
            # trace
            if accelerator is None and self.ipex is False:
                return model
            if accelerator in ("jit", None):
                acce_model = \
                    InferenceOptimizer.trace(model=model,
                                             accelerator=accelerator,
                                             use_ipex=self.ipex,
                                             # channels_last is only for jit
                                             channels_last=self.channels_last,
                                             input_sample=input_sample)
            else:
                acce_model = \
                    InferenceOptimizer.trace(model=model,
                                             accelerator=accelerator,
                                             input_sample=input_sample,
                                             thread_num=thread_num,
                                             # remove output of openvino
                                             logging=logging)
        else:
            # quantize
            ort_method: str = self.method
            acce_model = \
                InferenceOptimizer.quantize(model=deepcopy(model),
                                            precision=self.get_precision(),
                                            accelerator=accelerator,
                                            use_ipex=self.ipex,
                                            calib_dataloader=training_data,
                                            input_sample=input_sample,
                                            method=ort_method,
                                            thread_num=thread_num,
                                            sample_size=sample_size_for_pot,
                                            # remove output of openvino
                                            logging=logging)
        return acce_model


class InferenceOptimizer(BaseInferenceOptimizer):

    # acceleration method combinations, developers may want to register some new
    # combinations here
    ALL_INFERENCE_ACCELERATION_METHOD = \
        {
            "original": TorchAccelerationOption(),
            "fp32_ipex": TorchAccelerationOption(ipex=True),
            "bf16": TorchAccelerationOption(bf16=True),
            "bf16_ipex": TorchAccelerationOption(bf16=True, ipex=True),
            "int8": TorchAccelerationOption(inc=True),
            "jit_fp32": TorchAccelerationOption(jit=True),
            "jit_fp32_ipex": TorchAccelerationOption(jit=True, ipex=True),
            "jit_fp32_ipex_channels_last": TorchAccelerationOption(jit=True, ipex=True,
                                                                   channels_last=True),
            "openvino_fp32": TorchAccelerationOption(openvino=True),
            "openvino_int8": TorchAccelerationOption(openvino=True, pot=True),
            "onnxruntime_fp32": TorchAccelerationOption(onnxruntime=True),
            "onnxruntime_int8_qlinear": TorchAccelerationOption(onnxruntime=True, inc=True,
                                                                method="qlinear"),
            "onnxruntime_int8_integer": TorchAccelerationOption(onnxruntime=True, inc=True,
                                                                method="integer"),
        }

    def optimize(self, model: nn.Module,
                 training_data: Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]],
                 validation_data:
                     Optional[Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]]] = None,
                 input_sample: Union[torch.Tensor, Dict, Tuple[torch.Tensor], None] = None,
                 metric: Optional[Callable] = None,
                 direction: str = "max",
                 thread_num: Optional[int] = None,
                 logging: bool = False,
                 latency_sample_num: int = 100,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None) -> None:
        '''
        This function will give all available inference acceleration methods a try
        and record the latency, accuracy and model instance inside the Optimizer for
        future usage. All model instance is setting to eval mode.

        The available methods are "original", "fp32_ipex", "bf16", "bf16_ipex","int8",
        "jit_fp32", "jit_fp32_ipex", "jit_fp32_ipex_channels_last", "openvino_fp32",
        "openvino_int8", "onnxruntime_fp32", "onnxruntime_int8_qlinear"
        and "onnxruntime_int8_integer".

        :param model: A torch.nn.Module to be optimized
        :param training_data: training_data support following formats:

                | 1. a torch.utils.data.dataloader.DataLoader object for training dataset.
                | Users should be careful with this parameter since this dataloader
                | might be exposed to the model, which causing data leak. The
                | batch_size of this dataloader is important as well, users may
                | want to set it to the same batch size you may want to use the model
                | in real deploy environment. E.g. batch size should be set to 1
                | if you would like to use the accelerated model in an online service.
                |
                | 2. a single torch.Tensor which used for training, this case is used to
                | accept single sample input x.
                |
                | 3. a tuple of torch.Tensor which used for training, this case is used to
                | accept single sample input (x, y) or (x1, x2) et al.

        :param validation_data: (optional) validation_data is only needed when users care
                                about the possible accuracy drop. It support following formats:

                | 1. a torch.utils.data.dataloader.DataLoader object for accuracy evaluation.
                |
                | 2. a single torch.Tensor which used for training, this case is used to
                | accept single sample input x.
                |
                | 3. a tuple of torch.Tensor which used for training, this case is used to
                | accept single sample input (x, y) or (x1, x2) et al.

        :param input_sample: (optional) A set of inputs for trace, defaults to None.
               In most cases, you don't need specify this parameter, it will be obtained from
               training_data.
        :param metric: (optional) A callable object which is used for calculating accuracy.
               It supports two kinds of callable object:

               | 1. A torchmetrics.Metric object or similar callable object which takes
               | prediction and target then returns an accuracy value in this calling
               | method `metric(pred, target)`. This requires data in validation_data
               | is composed of (input_data, target).
               | 2. A callable object that takes model and validation_data (if
               | validation_data is not None) as input, and returns an accuracy value in
               | this calling method metric(model, data_loader) (or metric(model) if
               | validation_data is None).

        :param direction: (optional) A string that indicates the higher/lower
               better for the metric, "min" for the lower the better and "max" for the
               higher the better. Default value is "max".
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
               inference.
        :param logging: whether to log detailed information of model conversion.
               Default: False.
        :param latency_sample_num: (optional) a int represents the number of repetitions
               to calculate the average latency. The default value is 100.
        :param includes: (optional) a list of acceleration methods that will be included in the
               search. Default to None meaning including all available methods. "original" method
               will be automatically add to includes.
        :param excludes: (optional) a list of acceleration methods that will be excluded from the
               search. "original" will be ignored in the excludes.
        '''

        # check if model is a nn.Module or inherited from a nn.Module
        invalidInputError(isinstance(model, nn.Module), "model should be a nn module.")
        invalidInputError(direction in ['min', 'max'],
                          "Only support direction 'min', 'max'.")

        # get the available methods whose dep is met
        available_dict: Dict =\
            available_acceleration_combination(excludes=excludes,
                                               includes=includes,
                                               full_methods=self.ALL_INFERENCE_ACCELERATION_METHOD)

        self._direction: str = direction  # save direction as attr
        # record whether calculate accuracy in optimize by this attr
        if validation_data is None and metric is None:
            self._calculate_accuracy = False
        else:
            # test whether accuracy calculation works later
            self._calculate_accuracy = True

        default_threads: int = torch.get_num_threads()
        thread_num: int = default_threads if thread_num is None else int(thread_num)

        result_map: Dict[str, Dict] = {}

        model.eval()  # change model to eval mode

        if input_sample is None:
            forward_args = get_forward_args(model)
            if isinstance(training_data, DataLoader):
                input_sample = get_input_example(model, training_data, forward_args)
            else:
                if isinstance(training_data, Sequence):
                    input_sample = tuple(list(training_data)[:len(forward_args)])
                else:
                    input_sample = training_data
                # turn training_data into dataset
                dataset = RepeatDataset(sample=training_data, num=1)
                training_data = DataLoader(dataset, batch_size=1)
                training_data = remove_batch_dim_fn(training_data)
                if validation_data is not None and not isinstance(validation_data, DataLoader):
                    # turn validation_data into dataset
                    val_dataset = RepeatDataset(sample=validation_data, num=1)
                    validation_data = DataLoader(val_dataset, batch_size=1)
                    validation_data = remove_batch_dim_fn(validation_data)

        st = time.perf_counter()
        try:
            with torch.no_grad():
                if isinstance(input_sample, (Dict, torch.Tensor)):
                    model(input_sample)
                else:
                    model(*input_sample)
        except Exception:
            invalidInputError(False,
                              "training_data is incompatible with your model input.")
        baseline_time = time.perf_counter() - st
        if baseline_time > 0.1:  # 100ms
            sample_size_for_pot = 15
        else:
            sample_size_for_pot = 100

        print("==========================Start Optimization==========================")
        start_time = time.perf_counter()
        for idx, (method, available) in enumerate(available_dict.items()):
            result_map[method] = {}
            if available is False:
                result_map[method]["status"] = "lack dependency"
            else:
                print(f"----------Start test {method} model "
                      f"({idx+1}/{len(available_dict)})----------")
                option: AccelerationOption = self.ALL_INFERENCE_ACCELERATION_METHOD[method]
                precision = option.get_precision()
                try:
                    acce_model = option.optimize(model, training_data=training_data,
                                                 input_sample=input_sample,
                                                 thread_num=thread_num,
                                                 logging=logging,
                                                 sample_size_for_pot=sample_size_for_pot)
                except Exception as e:
                    traceback.print_exc()
                    result_map[method]["status"] = "fail to convert"
                    print(f"----------Failed to convert to {method}----------")
                    continue

                result_map[method]["status"] = "successful"

                def func_test(model, input_sample):
                    with torch.no_grad():
                        if isinstance(input_sample, (Dict, torch.Tensor)):
                            model(input_sample)
                        else:
                            model(*input_sample)

                torch.set_num_threads(thread_num)
                try:
                    result_map[method]["latency"], status =\
                        throughput_calculate_helper(latency_sample_num, baseline_time,
                                                    func_test, acce_model, input_sample)
                    if status is False and method != "original":
                        result_map[method]["status"] = "early stopped"
                        torch.set_num_threads(default_threads)
                        continue
                except Exception as e:
                    traceback.print_exc()
                    result_map[method]["status"] = "fail to forward"
                    print(f"----------{method} failed to forward----------")
                    torch.set_num_threads(default_threads)
                    continue

                torch.set_num_threads(default_threads)
                if self._calculate_accuracy:
                    # here we suppose trace don't change accuracy,
                    # so we jump it to reduce time cost of optimize
                    if precision == "fp32" and method != "original":
                        result_map[method]["accuracy"] = "not recomputed"
                    else:
                        if method == "original":
                            # test whether metric works
                            try:
                                result_map[method]["accuracy"] =\
                                    _accuracy_calculate_helper(acce_model, metric,
                                                               validation_data)
                            except Exception as e:
                                traceback.print_exc()
                                self._calculate_accuracy = False
                                invalidInputError(
                                    False,
                                    "Your metric is incompatible with validation_data or don't "
                                    "follow our given pattern. Our expected metric pattern is "
                                    "as follows:\n1. a torchmetrics.Metric object\n2. a callable "
                                    "object which takes prediction and target then returns a value"
                                    " in this calling method `metric(pred, target)`\n3. a callable"
                                    " object that takes model and validation_data (if "
                                    "validation_data is not None) as input, and returns an accuracy"
                                    " value in this calling method metric(model, data_loader) "
                                    "(or metric(model) if validation_data is None).")
                        else:
                            result_map[method]["accuracy"] =\
                                _accuracy_calculate_helper(acce_model, metric,
                                                           validation_data)
                else:
                    result_map[method]["accuracy"] = None

                result_map[method]["model"] = acce_model
                print(f"----------Finish test {method} model "
                      f"({idx+1}/{len(available_dict)})----------")

        self.optimized_model_dict: Dict = result_map
        print("\n\n==========================Optimization Results==========================")

        self._optimize_result = format_optimize_result(self.optimized_model_dict,
                                                       self._calculate_accuracy)
        # save time cost to self._optimize_result
        time_cost = time.perf_counter() - start_time
        time_cost_str = f"Optimization cost {time_cost:.1f}s in total."
        self._optimize_result += time_cost_str
        print(self._optimize_result)
        print("===========================Stop Optimization===========================")

    @staticmethod
    def quantize(model: nn.Module,
                 precision: str = 'int8',
                 accelerator: Optional[str] = None,
                 use_ipex: bool = False,
                 calib_dataloader: Optional[DataLoader] = None,
                 metric: Optional[Metric] = None,
                 accuracy_criterion: Optional[dict] = None,
                 approach: str = 'static',
                 method: Optional[str] = None,
                 conf: Optional[str] = None,
                 tuning_strategy: Optional[str] = None,
                 timeout: Optional[int] = None,
                 max_trials: Optional[int] = None,
                 input_sample=None,
                 thread_num: Optional[int] = None,
                 onnxruntime_session_options=None,
                 openvino_config=None,
                 simplification: bool = True,
                 sample_size: int = 100,
                 logging: bool = True,
                 **export_kwargs):
        """
        Calibrate a torch.nn.Module for post-training quantization.

        :param model:           A model to be quantized. Model type should be an instance of
                                torch.nn.Module.
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
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
        :param sample_size: (optional) a int represents how many samples will be used for
                            Post-training Optimization Tools (POT) from OpenVINO toolkit,
                            only valid for accelerator='openvino'. Default to 100.
                            The larger the value, the more accurate the conversion,
                            the lower the performance degradation, but the longer the time.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return:            A accelerated torch.nn.Module if quantization is sucessful.
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
            # transform the dataloader to inc mode
            inc_calib_dataloader =\
                transform_multiple_input_dataloader_to_inc_mode(model,
                                                                calib_dataloader)
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
                        if onnxruntime_session_options is None:
                            import onnxruntime
                            onnxruntime_session_options = onnxruntime.SessionOptions()
                            if thread_num is not None:
                                onnxruntime_session_options.intra_op_num_threads = thread_num
                                onnxruntime_session_options.inter_op_num_threads = thread_num
                        model = InferenceOptimizer.trace(
                            model,
                            input_sample=input_sample,
                            accelerator='onnxruntime',
                            onnxruntime_session_options=onnxruntime_session_options,
                            simplification=simplification,
                            **export_kwargs)
                """
                If accelerator==None, quantized model returned should be an object of PytorchModel
                which is defined by neural-compressor containing a `GraphModule` for inference.
                Otherwise accelerator=='onnxruntime', it returns an ONNXModel object. A supported
                model which is able to run on Pytorch or ONNXRuntime can be fetched by
                `quantized_model.model`.
                """
                return inc_quantize(model, inc_calib_dataloader, metric,
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
                                                     thread_num=thread_num,
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
                    "sample_size": sample_size
                }
                return model.pot(calib_dataloader, thread_num=thread_num,
                                 config=openvino_config, **kwargs)
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid.".format(accelerator))
        invalidInputError(False,
                          "Precision {} is invalid.".format(precision))

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator: Optional[str] = None,
              use_ipex: bool = False,
              thread_num: Optional[int] = None,
              onnxruntime_session_options=None,
              openvino_config=None,
              simplification: bool = True,
              logging: bool = True,
              **export_kwargs):
        """
        Trace a torch.nn.Module and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param model: An torch.nn.Module model, including pl.LightningModule.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch
                            backend. 'openvino', 'onnxruntime' and 'jit' are supported for now.
        :param use_ipex: whether we use ipex as accelerator for inferencing. default: False.
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :param **kwargs: other extra advanced settings include
                         1. those be passed to torch.onnx.export function, only valid when
                         accelerator='onnxruntime'/'openvino', otherwise will be ignored.
                         2. if channels_last is set and `use_ipex=True`, we will transform the
                         data to be channels last according to the setting. Defaultly, channels_last
                         will be set to ``True`` if `use_ipex=True`.
        :return: Model with different acceleration.
        """
        invalidInputError(
            isinstance(model, nn.Module) and not isinstance(model, AcceleratedLightningModule),
            "Expect a nn.Module instance that is not traced or quantized"
            "but got type {}".format(type(model))
        )
        if accelerator == 'openvino':  # openvino backend will not care about ipex usage
            final_openvino_option = {"INFERENCE_PRECISION_HINT": "f32"}
            if openvino_config is not None:
                final_openvino_option.update(openvino_config)
            return PytorchOpenVINOModel(model, input_sample, thread_num, logging,
                                        final_openvino_option, **export_kwargs)
        if accelerator == 'onnxruntime':  # onnxruntime backend will not care about ipex usage
            if onnxruntime_session_options is None:
                import onnxruntime
                onnxruntime_session_options = onnxruntime.SessionOptions()
                if thread_num is not None:
                    onnxruntime_session_options.intra_op_num_threads = thread_num
                    onnxruntime_session_options.inter_op_num_threads = thread_num
            return PytorchONNXRuntimeModel(model, input_sample, onnxruntime_session_options,
                                           simplification=simplification, **export_kwargs)
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

    @staticmethod
    def save(model: nn.Module, path):
        """
        Save the model to local file.

        :param model: Any model of torch.nn.Module, including all models accelareted by
               Trainer.trace/Trainer.quantize.
        :param path: Path to saved model. Path should be a directory.
        """
        save_model(model, path)

    @staticmethod
    def load(path, model: Optional[nn.Module] = None):
        """
        Load a model from local.

        :param path: Path to model to be loaded. Path should be a directory.
        :param model: Required FP32 model to load pytorch model, it is needed if you accelerated
               the model with accelerator=None by Trainer.trace/Trainer.quantize. model
               should be set to None if you choose accelerator="onnxruntime"/"openvino"/"jit".
        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or
                 precision(FP32/FP16/BF16/INT8).
        """
        return load_model(path, model)

    @staticmethod
    def to_multi_instance(model: nn.Module, num_processes: int) -> _MultiInstanceModel:
        """
        Transform a model to multi-instance inference model.
        :param model: The model to transform.
        :param num_processes: The number of processes which will be used.
        :return: Model with multi-instance inference acceleration.
        """
        p_num = num_processes
        mgr = mp.Manager()
        send_queues = [mgr.Queue() for _ in range(p_num)]
        recv_queues = [mgr.Queue() for _ in range(p_num)]

        KMP_AFFINITY = os.environ.get("KMP_AFFINITY", "")
        OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "")
        envs = schedule_processors(p_num)
        ps = []
        for i in range(p_num):
            os.environ["KMP_AFFINITY"] = envs[i]['KMP_AFFINITY']
            os.environ["OMP_NUM_THREADS"] = envs[i]['OMP_NUM_THREADS']

            p = mp.Process(target=_multi_instance_helper,
                           args=(model, send_queues[i], recv_queues[i]))
            p.start()
            ps.append(p)
        os.environ["KMP_AFFINITY"] = KMP_AFFINITY
        os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS

        return _MultiInstanceModel(model, ps, mgr, send_queues, recv_queues)


def _signature_check(function):
    '''
    A quick helper to judge whether input function is following this calling
    method `metric(pred, target)`.
    '''
    import inspect
    sig = inspect.signature(function)
    if len(sig.parameters.values()) < 2:
        return False
    param1_name = list(sig.parameters.values())[0].name
    param2_name = list(sig.parameters.values())[1].name
    if "pred" in param1_name and "target" in param2_name:
        return True
    return False


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    if isinstance(metric, Metric) or _signature_check(metric) is True:
        invalidInputError(data is not None,
                          "Validation data can't be None when you pass a "
                          "torchmetrics.Metric object or similar callable "
                          "object which takes prediction and target as input.")
        metric = NanoMetric(metric)
        return metric(model, data)
    else:
        if data is None:
            return metric(model)
        else:
            return metric(model, data)
