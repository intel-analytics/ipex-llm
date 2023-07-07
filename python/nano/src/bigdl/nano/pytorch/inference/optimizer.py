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
import sigfig
import multiprocessing as mp
from typing import Dict, Callable, Tuple, Optional, List, Union, Sequence, Mapping
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from bigdl.nano.utils.common import AccelerationOption, available_acceleration_combination,\
    latency_calculate_helper, torch_loader_latency_calculate_helper,\
    format_optimize_result, BaseInferenceOptimizer
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.amp import BF16Model
from bigdl.nano.pytorch.low_precision.jit_int8_api import PytorchJITINT8Model
from bigdl.nano.deps.openvino.openvino_api import PytorchOpenVINOModel
from bigdl.nano.deps.ipex.ipex_api import PytorchIPEXJITModel, PytorchIPEXJITBF16Model,\
    PytorchIPEXQuantizationModel, PytorchIPEXPUModel
from bigdl.nano.deps.onnxruntime.onnxruntime_api import PytorchONNXRuntimeModel
from bigdl.nano.deps.neural_compressor.inc_api import quantize as inc_quantize
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.pytorch import get_forward_args, get_input_example
from bigdl.nano.utils.pytorch import NanoMetric
from bigdl.nano.utils.pytorch import RepeatDataset, remove_batch_dim_fn
from bigdl.nano.utils.pytorch import transform_multiple_input_dataloader_to_inc_mode,\
    automatic_add_label_in_dataloader
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_10
from bigdl.nano.utils.pytorch import save_model, load_model
from bigdl.nano.utils.common import schedule_processors
from bigdl.nano.utils.common import EnvContext
from bigdl.nano.pytorch.context_manager import generate_context_manager,\
    BaseContextManager, AutocastContextManager
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
                 thread_num=None, dynamic_axes=True, logging=False,
                 sample_size_for_pot=100, output_tensors=True,
                 jit_strict=True):
        accelerator = self.get_accelerator()
        if self.get_precision() == "fp32":
            if accelerator is None and self.ipex is False and \
                    self.channels_last is False:
                return model
            # trace
            acce_model = \
                InferenceOptimizer.trace(model=model,
                                         accelerator=accelerator,
                                         input_sample=input_sample,
                                         thread_num=thread_num,
                                         channels_last=self.channels_last,
                                         use_ipex=self.ipex,
                                         dynamic_axes=dynamic_axes,
                                         # remove output of openvino
                                         logging=logging,
                                         output_tensors=output_tensors,
                                         jit_strict=jit_strict)
        else:
            # quantize
            ort_method: str = self.method
            acce_model = \
                InferenceOptimizer.quantize(model=model,
                                            precision=self.get_precision(),
                                            accelerator=accelerator,
                                            use_ipex=self.ipex,
                                            channels_last=self.channels_last,
                                            calib_data=training_data,
                                            input_sample=input_sample,
                                            method=ort_method,
                                            thread_num=thread_num,
                                            dynamic_axes=dynamic_axes,
                                            sample_size=sample_size_for_pot,
                                            # remove output of openvino
                                            logging=logging,
                                            output_tensors=output_tensors,
                                            jit_strict=jit_strict)
        return acce_model


class InferenceOptimizer(BaseInferenceOptimizer):

    # acceleration method combinations, developers may want to register some new
    # combinations here
    ALL_INFERENCE_ACCELERATION_METHOD = \
        {
            "original": TorchAccelerationOption(),
            "fp32_channels_last": TorchAccelerationOption(channels_last=True),
            "fp32_ipex": TorchAccelerationOption(ipex=True),
            "fp32_ipex_channels_last": TorchAccelerationOption(ipex=True,
                                                               channels_last=True),
            "bf16": TorchAccelerationOption(bf16=True),
            "bf16_channels_last": TorchAccelerationOption(bf16=True,
                                                          channels_last=True),
            "bf16_ipex": TorchAccelerationOption(bf16=True, ipex=True),
            "bf16_ipex_channels_last": TorchAccelerationOption(bf16=True, ipex=True,
                                                               channels_last=True),
            "static_int8": TorchAccelerationOption(inc=True),
            "static_int8_ipex": TorchAccelerationOption(inc=True, method="ipex",
                                                        ipex=True),
            "jit_fp32": TorchAccelerationOption(jit=True),
            "jit_fp32_channels_last": TorchAccelerationOption(jit=True,
                                                              channels_last=True),
            "jit_bf16": TorchAccelerationOption(jit=True, bf16=True),
            "jit_bf16_channels_last": TorchAccelerationOption(jit=True, bf16=True,
                                                              channels_last=True),
            "jit_fp32_ipex": TorchAccelerationOption(jit=True, ipex=True),
            "jit_fp32_ipex_channels_last": TorchAccelerationOption(jit=True, ipex=True,
                                                                   channels_last=True),
            "jit_bf16_ipex": TorchAccelerationOption(jit=True, bf16=True, ipex=True),
            "jit_bf16_ipex_channels_last": TorchAccelerationOption(jit=True, bf16=True,
                                                                   ipex=True,
                                                                   channels_last=True),
            "jit_int8": TorchAccelerationOption(fx=True, jit=True),
            "jit_int8_channels_last": TorchAccelerationOption(fx=True, jit=True,
                                                              channels_last=True),
            "openvino_fp32": TorchAccelerationOption(openvino=True),
            "openvino_bf16": TorchAccelerationOption(openvino=True, bf16=True),
            "openvino_fp16": TorchAccelerationOption(openvino=True, fp16=True),
            "openvino_int8": TorchAccelerationOption(openvino=True, pot=True),
            "onnxruntime_fp32": TorchAccelerationOption(onnxruntime=True),
            "onnxruntime_int8_qlinear": TorchAccelerationOption(onnxruntime=True, inc=True,
                                                                method="qlinear"),
            "onnxruntime_int8_integer": TorchAccelerationOption(onnxruntime=True, inc=True,
                                                                method="integer"),
        }

    _default_methods = ["original", "bf16", "static_int8",
                        "jit_fp32_ipex", "jit_fp32_ipex_channels_last",
                        "jit_bf16_ipex", "jit_bf16_ipex_channels_last", "openvino_fp32",
                        "openvino_int8", "onnxruntime_fp32", "onnxruntime_int8_qlinear"]
    DEFAULT_INFERENCE_ACCELERATION_METHOD = {}
    for method in _default_methods:
        DEFAULT_INFERENCE_ACCELERATION_METHOD[method] = ALL_INFERENCE_ACCELERATION_METHOD[method]

    def optimize(self, model: nn.Module,
                 training_data: Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]],
                 validation_data:
                     Optional[Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]]] = None,
                 input_sample: Union[torch.Tensor, Dict, Tuple[torch.Tensor], None] = None,
                 metric: Optional[Callable] = None,
                 direction: str = "max",
                 thread_num: Optional[int] = None,
                 accelerator: Optional[Tuple[str]] = None,
                 precision: Optional[Tuple[str]] = None,
                 use_ipex: Optional[bool] = None,
                 jit_strict: Optional[bool] = True,
                 enable_onednn: Optional[bool] = False,
                 search_mode: str = "default",
                 dynamic_axes: Union[bool, dict] = True,
                 logging: bool = False,
                 output_tensors: bool = True,
                 latency_sample_num: int = 100,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None,
                 output_filename: Optional[str] = None,
                 no_cache: bool = False) -> None:
        '''
        This function will give all available inference acceleration methods a try
        and record the latency, accuracy and model instance inside the Optimizer for
        future usage. All model instance is setting to eval mode.

        The available methods are "original", "fp32_channels_last", "fp32_ipex",
        "fp32_ipex_channels_last", "bf16", "bf16_channels_last", "bf16_ipex",
        "bf16_ipex_channels_last", "static_int8", "static_int8_ipex", "jit_fp32",
        "jit_fp32_channels_last", "jit_bf16", "jit_bf16_channels_last",
        "jit_fp32_ipex", "jit_fp32_ipex_channels_last", "jit_bf16_ipex",
        "jit_bf16_ipex_channels_last", "jit_int8", "jit_int8_channels_last",
        "openvino_fp32", "openvino_int8", "onnxruntime_fp32",
        "onnxruntime_int8_qlinear" and "onnxruntime_int8_integer".

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
                | Each element in the DataLoader can be one of the following:
                |    a. a single Tensor or dict of Tensors
                |    b. a tuple:
                |         b1: if the length is 1, the first element will be treated as input
                |             to the model
                |         b2: if the length is 2, the first element will be treated as input
                |             to the model, with the sencond element treated as label.
                |             if the input to the model is a tuple, it will be unpacked as
                |             multiple inputs.
                |         b3: if the length is larger than 2, the first n elements as input
                |             to the model, with n being the argument lenth to the model.forward
                |             and the rest will be treated as label
                |
                | 2. a single element of the Dataloader specified above

        :param validation_data: (optional) validation_data is only needed when users care
                                about the possible accuracy drop. It support following formats:

                | 1. a torch.utils.data.dataloader.DataLoader object for accuracy evaluation.
                |
                | Each element in the DataLoader should be a tuple as least size of two:
                |     a: if the length is 2, the first element will be treated as input
                |        to the model, with the sencond element treated as label
                |     b: if the length is larger than 2, the first n elements as input
                |        to the model, with n being the argument lenth to the model.forward
                |        and the rest will be treated as label
                |
                | 2. a single element of the Dataloader specified above

        :param input_sample: (optional) A set of inputs for trace, defaults to None.
               In most cases, you don't need specify this parameter, it will be obtained from
               training_data. You have to specify this parameter only if the forward function
               of your model contains some kwargs like `def forward(self, x1, x2, x3=1)`.
        :param metric: (optional) A callable object which is used for calculating accuracy.
               It supports two kinds of callable object:

               | 1. A torchmetrics.Metric object or similar callable object which takes
               | prediction and target then returns an accuracy value in this calling
               | method `metric(pred, target)`. This requires data in validation_data
               | is composed of (input_data, target).
               |
               | 2. A callable object that takes model and validation_data (if
               | validation_data is not None) as input, and returns an accuracy value in
               | this calling method metric(model, data_loader) (or metric(model) if
               | validation_data is None). Note that there is no need to call `with
               | InferenceOptimizer.get_context()` in this object.

        :param direction: (optional) A string that indicates the higher/lower
               better for the metric, "min" for the lower the better and "max" for the
               higher the better. Default value is "max".
        :param thread_num: (optional) An int represents how many threads(cores) is needed for
               inference. This parameter only controls the usage of thread number in the process
               of latency calculation as well as later inference process of your obtained
               accelerated model. In other words, the process of model conversion and optional
               accuracy calculation won't be restricted by this parameter. Defaults to None,
               represents that all cores will be used.
        :param accelerator: (optional) A string tuple that specifies the accelerators to search.
               The optional accelerators are: None, 'openvino', 'onnxruntime', 'jit'.
               Defaults to None which represents there is no restriction on accelerators.
               If not None, then will only traverse corresponding methods whose accelerator falls
               within the specified accelerator tuple.
        :param precision: (optional) A string tuple that specifies the precision to search.
               The optional precision are: 'int8', 'bf16', and 'fp32'. Defaults to None which
               represents no precision limit. If not None, then will only traverse corresponding
               methods whose precision falls within the specified precision tuple.
        :param use_ipex: (optional) if not None, then will only try methods with/without
               this specific ipex setting.
        :param jit_strict: Whether recording your mutable container types. This parameter will be
               passed to ``torch.jit.trace``. if ``accelerator != 'jit'`` or
               ``jit_method='script'``, it will be ignored. Default to True.
        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph API,
                which provides a flexible API for aggressive fusion. Default to
                ``False``, only valid when accelerator='jit', otherwise will
                be ignored. For more details, please refer https://github.com/
                pytorch/pytorch/tree/master/torch/csrc/jit/codegen/
                onednn#pytorch---onednn-graph-api-bridge.
        :param search_mode: Here are three modes for optimization:

               | 1. default: This mode only traverses a subset of all combinations. This subset
               | is a collection of methods that we select based on experience and think have
               | better acceleration effect in general. This mode allows you to quickly obtain a
               | good acceleration method, but it is not necessarily the global optimal. Default
               | to this mode if you don't specify accelerator/precision/use_ipex.
               |
               | 2. all: This mode will traverse all possible combinations, which can ensure
               | find the global optimization, but it will take a long time.
               |
               | 3. grid: If you have specified accelerator/precision/use_ipex, the default is
               | grid mode. We will sort and combine according to the value you specified to
               | get the search range.

        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model
               will have the first dim of each Tensor input as a dynamic batch_size. If
               dynamic_axes=False, the exported model will have the shapes of all input and output
               tensors set to exactly match those given in input_sample. To specify axes of
               tensors as dynamic (i.e. known only at run-time), set dynamic_axes to a dict with
               schema:

               | KEY (str): an input or output name. Each name must also be provided
               | in input_names or output_names.
               |
               | VALUE (dict or list): If a dict, keys are axis indices and values are
               | axis names. If a list, each element is an axis index.

               If accelerator != 'openvino'/'onnxruntime', it will be ignored.
        :param logging: whether to log detailed information of model conversion.
               Default: False.
        :param output_tensors: boolean, default to True and output of the model will be Tensors,
                               only valid when accelerator='onnxruntime' or accelerator='openvino',
                               otherwise will be ignored. If output_tensors=False, output of the
                               export model will be ndarray.
        :param latency_sample_num: (optional) a int represents the number of repetitions
               to calculate the average latency. The default value is 100.
        :param includes: (optional) a list of acceleration methods that will be included in the
               search. Default to None meaning including all available methods. "original" method
               will be automatically add to includes.
        :param excludes: (optional) a list of acceleration methods that will be excluded from the
               search. "original" will be ignored in the excludes.
        :param output_filename: (optional) a string filename is used to specify the file which the
               optimized table will be writed. The default is None which means don't write to file.
        :param no_cache: if set True, calculate average latency by iterating all the samples from
               the provided dataloader until reaching the latency_sample_num. Default set to be
               False, meaning always loading one single sample from cache to test latency.
        '''

        # check if model is a nn.Module or inherited from a nn.Module
        invalidInputError(isinstance(model, nn.Module), "model should be a nn module.")
        invalidInputError(direction in ['min', 'max'],
                          "Only support direction 'min', 'max'.")
        invalidInputError(accelerator is None or isinstance(accelerator, tuple),
                          "accelerator must be a tuple.")
        invalidInputError(precision is None or isinstance(precision, tuple),
                          "precison must be a tuple.")
        _check_accelerator = accelerator is None or all(
            ac in [None, 'onnxruntime', 'openvino', 'jit'] for ac in accelerator)
        invalidInputError(_check_accelerator is True,
                          "Only support accelerator None, 'onnxruntime', 'openvino' and 'jit'.")
        _check_precision = precision is None or all(
            p in [None, 'int8', 'bf16', 'fp32'] for p in precision)
        invalidInputError(_check_precision is True,
                          "Only support precision 'int8', 'bf16', 'fp32'.")

        if accelerator is not None or precision is not None or use_ipex is not None:
            search_mode = "grid"
            # setting search scope
            all_acceleration_methods = _obtain_combinations(self.ALL_INFERENCE_ACCELERATION_METHOD,
                                                            precision,
                                                            accelerator,
                                                            use_ipex)
        else:
            if search_mode == "all":
                all_acceleration_methods = self.ALL_INFERENCE_ACCELERATION_METHOD
            elif search_mode == "default":
                # which is seting based on experience, and may need periodic update
                all_acceleration_methods = self.DEFAULT_INFERENCE_ACCELERATION_METHOD

        # get the available methods whose dep is met
        available_dict: Dict =\
            available_acceleration_combination(excludes=excludes,
                                               includes=includes,
                                               full_methods=all_acceleration_methods,
                                               all_methods=self.ALL_INFERENCE_ACCELERATION_METHOD)

        self._direction: str = direction  # save direction as attr
        # record whether calculate accuracy in optimize by this attr
        if validation_data is None and metric is None:
            self._calculate_accuracy = False
        else:
            # test whether accuracy calculation works later
            self._calculate_accuracy = True

        default_threads: int = torch.get_num_threads()
        thread_num: int = None if thread_num is None else int(thread_num)

        result_map: Dict[str, Dict] = {}

        model.eval()  # change model to eval mode

        forward_args = get_forward_args(model)
        if input_sample is None:
            if isinstance(training_data, DataLoader):
                input_sample = get_input_example(model, training_data, forward_args)
            else:
                if isinstance(training_data, Sequence):
                    if len(training_data) <= 2:
                        input_sample = training_data[0]
                        if len(training_data) == 2:
                            input_label = training_data[1]
                        else:
                            input_label = torch.Tensor([])
                    else:
                        input_sample = tuple(training_data[:len(forward_args)])
                        input_label = tuple(training_data[len(forward_args):])
                else:
                    input_sample = training_data
                    input_label = torch.Tensor([])
                # turn training_data into dataset
                dataset = RepeatDataset(sample=(input_sample, input_label), num=1)
                training_data = DataLoader(dataset, batch_size=1)
                training_data = remove_batch_dim_fn(training_data)

                if validation_data is not None and not isinstance(validation_data, DataLoader):
                    # turn validation_data into dataset
                    if isinstance(validation_data, Sequence):
                        if len(validation_data) <= 2:
                            val_sample = validation_data[0]
                            if len(validation_data) == 2:
                                val_label = validation_data[1]
                            else:
                                val_label = []
                        else:
                            val_sample = tuple(validation_data[:len(forward_args)])
                            val_label = tuple(validation_data[len(forward_args):])
                    else:
                        val_sample = training_data
                        val_label = []
                    val_dataset = RepeatDataset(sample=(val_sample, val_label), num=1)
                    validation_data = DataLoader(val_dataset, batch_size=1)
                    validation_data = remove_batch_dim_fn(validation_data)
        # jit cannot handle `Mapping`, so we convert it to `dict`
        if isinstance(input_sample, Mapping):
            input_sample = dict(input_sample)
        st = time.perf_counter()
        try:
            with torch.no_grad():
                if isinstance(input_sample, (list, tuple)):
                    model(*input_sample)
                else:
                    if not isinstance(input_sample, (dict, torch.Tensor)):
                        warnings.warn("You may need to change `input_sample` to "
                                      "a (list/tuple/dict of) Tensor to use jit.")
                    model(input_sample)
        except Exception:
            invalidInputError(False,
                              f"training_data is incompatible with your model input.")
        baseline_time = time.perf_counter() - st
        if baseline_time > 0.1:  # 100ms
            sample_size_for_pot = 15
        else:
            sample_size_for_pot = 100

        # patch context manager
        model._nano_context_manager = generate_context_manager(accelerator=None,
                                                               precision="fp32",
                                                               thread_num=thread_num)

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
                _precision = option.get_precision()
                try:
                    acce_model = option.optimize(model,
                                                 training_data=training_data,
                                                 input_sample=input_sample,
                                                 thread_num=thread_num,
                                                 dynamic_axes=dynamic_axes,
                                                 logging=logging,
                                                 output_tensors=output_tensors,
                                                 sample_size_for_pot=sample_size_for_pot,
                                                 jit_strict=jit_strict)
                except Exception:
                    traceback.print_exc()
                    result_map[method]["status"] = "fail to convert"
                    print(f"----------Failed to convert to {method}----------")
                    continue

                result_map[method]["status"] = "successful"

                def func_test(model, input_sample):
                    if isinstance(input_sample, (list, tuple)):
                        model(*input_sample)
                    else:
                        model(input_sample)

                with InferenceOptimizer.get_context(acce_model):
                    try:
                        if no_cache:
                            result_map[method]["latency"], status =\
                                torch_loader_latency_calculate_helper(latency_sample_num,
                                                                      baseline_time,
                                                                      func_test,
                                                                      acce_model,
                                                                      input_sample,
                                                                      training_data,
                                                                      forward_args)
                        else:
                            result_map[method]["latency"], status =\
                                latency_calculate_helper(latency_sample_num, baseline_time,
                                                         func_test, acce_model, input_sample)
                        if status is False and method != "original":
                            result_map[method]["status"] = "early stopped"
                            # save model even early stop
                            result_map[method]["model"] = acce_model
                            torch.set_num_threads(default_threads)
                            continue
                    except Exception:
                        traceback.print_exc()
                        result_map[method]["status"] = "fail to forward"
                        print(f"----------{method} failed to forward----------")
                        torch.set_num_threads(default_threads)
                        continue

                    torch.set_num_threads(default_threads)
                    if self._calculate_accuracy:
                        # here we suppose trace don't change accuracy,
                        # so we jump it to reduce time cost of optimize
                        if _precision == "fp32" and method != "original":
                            _accuracy = result_map["original"]["accuracy"]
                            if isinstance(_accuracy, torch.Tensor):
                                _accuracy = _accuracy.item()
                            _accuracy = sigfig.round(_accuracy, sigfigs=5)
                            result_map[method]["accuracy"] = str(_accuracy) + '*'
                        else:
                            if method == "original":
                                # test whether metric works
                                try:
                                    result_map[method]["accuracy"] =\
                                        _accuracy_calculate_helper(acce_model, metric,
                                                                   validation_data)
                                except Exception:
                                    traceback.print_exc()
                                    self._calculate_accuracy = False
                                    invalidInputError(
                                        False,
                                        "Your metric is incompatible with validation_data or don't "
                                        "follow our given pattern. Our expected metric pattern is "
                                        "as follows:\n1. a torchmetrics.Metric object\n2. a "
                                        "callable object which takes prediction and target then "
                                        "returns a value in this calling method: `metric(pred, "
                                        "target)`\n3. a callable object that takes model and "
                                        "validation_data (if validation_data is not None) as input,"
                                        "and returns an accuracy value in this calling method: "
                                        "metric(model, data_loader) (or metric(model) if "
                                        "validation_data is None).")
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
        if self._calculate_accuracy:
            if precision is None or 'fp32' in precision:
                # only show this line when there is traced model and metric value
                self._optimize_result += "* means we assume the metric value of the traced "\
                    "model does not change, so we don't recompute metric value to save time.\n"
        # save time cost to self._optimize_result
        time_cost = time.perf_counter() - start_time
        time_cost_str = f"Optimization cost {time_cost:.1f}s in total."
        self._optimize_result += time_cost_str
        if output_filename is not None:
            with open(output_filename, "w") as f:
                f.write(self._optimize_result)
        print(self._optimize_result)
        print("===========================Stop Optimization===========================")

    @staticmethod
    def quantize(model: nn.Module,
                 precision: str = 'int8',
                 accelerator: Optional[str] = None,
                 use_ipex: bool = False,
                 calib_data: Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]] = None,
                 calib_dataloader: Union[DataLoader] = None,
                 eval_func: Optional[Callable] = None,
                 metric: Optional[Metric] = None,
                 accuracy_criterion: Optional[dict] = None,
                 approach: str = 'static',
                 method: Optional[str] = None,
                 conf: Optional[str] = None,
                 tuning_strategy: Optional[str] = None,
                 timeout: Optional[int] = None,
                 max_trials: Optional[int] = None,
                 input_sample=None,
                 channels_last: bool = False,
                 thread_num: Optional[int] = None,
                 device: Optional[str] = 'CPU',
                 onnxruntime_session_options=None,
                 openvino_config=None,
                 simplification: bool = True,
                 jit_strict: bool = True,
                 jit_method: Optional[str] = None,
                 dynamic_axes: Union[bool, dict] = True,
                 sample_size: int = 100,
                 logging: bool = True,
                 inplace: bool = False,
                 weights_prepack: Optional[bool] = None,
                 enable_onednn: bool = False,
                 q_config=None,
                 output_tensors: bool = True,
                 example_kwarg_inputs=None,
                 **kwargs):
        """
        Calibrate a torch.nn.Module for post-training quantization.

        :param model:           A model to be quantized. Model type should be an instance of
                                torch.nn.Module.
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', 'bf16', 'fp16', defaults to 'int8'.
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', 'jit', defaults
                                to None. None means staying in pytorch.
        :param use_ipex:        Whether we use ipex as accelerator for inference.
                                If precision != bf16, it will be ignored. Default: ``False``.
        :param calib_data:      Calibration data is required for static quantization.
                                It's also used as validation dataloader.
                                calib_data support following formats:

                                | 1. a torch.utils.data.dataloader.DataLoader object for training.
                                |
                                | 2. a single torch.Tensor used for training, this case is used
                                | to accept single sample input x.
                                |
                                | 3. a tuple of torch.Tensor which used for training, this case is
                                | used to accept single sample input (x, y) or (x1, x2) et al.
        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.
                                    Required for static quantization.
                                    It's also used as validation dataloader.

               .. warning::
                  ``calib_dataloader`` will be deprecated in future release.

                  Please use ``calib_data`` instead.
        :param eval_func:       A evaluation function which only accepts model as input and return
                                evaluation value. This parameter provides a higher degree of
                                freedom than using eval_loader and metric. Default to None meaning
                                no performance tuning, but it would be better give an evaluation
                                function to get better quantization performance.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param accuracy_criterion:  Tolerable accuracy drop, defaults to None meaning no
                                    accuracy control.
                                    accuracy_criterion = {'absolute':0.99, 'higher_is_better':False}
                                    means accuracy loss must be smaller than 0.99. For example, if
                                    higher_is_better is True, then this requires original metric
                                    value subtract current metric value be smaller than 0.99.
                                    For inc 1.x, this value must be set to [0, 1), for inc 2.x,
                                    there is no limit.
                                    accuracy_criterion = {'relative':0.1, 'higher_is_better':True}
                                    allows relative accuracy loss: 10%.
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
        :param input_sample:      An input example to convert pytorch model into ONNX/OpenVINO/JIT.
        :param channels_last: Whether use channels last memory format, i.e. NHWC (batch size,
                              height, width, channels), as an alternative way to store tensors in
                              classic/contiguous NCHW order, only valid when precision='bf16',
                              otherwise will be ignored. This setting only works for 4-dim Tensor.
                              Default: ``False``.
        :param thread_num: (optional) An int represents how many threads(cores) is needed for
                           inference. This parameter only controls the usage of thread number in
                           later inference process of your obtained accelerated model. In other
                           words, the process of model conversion won't be restricted by this
                           parameter.
        :param device: (optional) A string represents the device of the inference. Default to 'CPU',
                        only valid when accelerator='openvino', otherwise will be ignored.
                        'CPU', 'GPU' and 'VPUX' are supported for now.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
        :param jit_strict: Whether recording your mutable container types. This parameter will be
                           passed to ``torch.jit.trace``. if ``accelerator != 'jit'`` or
                           ``jit_method='script'``, it will be ignored. Default to True.
        :param jit_method: Whether to use ``jit.trace`` or ``jit.script`` to convert a model
                           to TorchScript. Accepted values are ``'trace'``, ``'script'``,
                           and ``None``. Default to be ``None`` meaning the try-except logic
                           to use ``jit.trace`` or ``jit.script``. If ``accelerator != 'jit'``,
                           this parameter will be ignored.
        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model
                             will have the first dim of each Tensor input as a dynamic batch_size.
                             If dynamic_axes=False, the exported model will have the shapes of all
                             input and output tensors set to exactly match those given in
                             input_sample. To specify axes of tensors as dynamic (i.e. known only
                             at run-time), set dynamic_axes to a dict with schema:

                             | KEY (str): an input or output name. Each name must also be provided
                             | in input_names or output_names.
                             |
                             | VALUE (dict or list): If a dict, keys are axis indices and values
                             | are axis names. If a list, each element is an axis index.

                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.
        :param sample_size: (optional) a int represents how many samples will be used for
                            Post-training Optimization Tools (POT) from OpenVINO toolkit,
                            only valid for accelerator='openvino'. Default to 100.
                            The larger the value, the more accurate the conversion,
                            the lower the performance degradation, but the longer the time.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :param inplace: whether to perform inplace optimization. Default: ``False``.
        :param weights_prepack: Whether to perform weight prepack for convolution and linear to
                                avoid oneDNN weights reorder. The default value is None. Explicitly
                                setting this knob overwrites the configuration set by level knob.
                                Only valid when ``use_ipex=True``, otherwise will be ignored.
                                You can try to reduce the occupied memory size by setting this
                                parameter to ``False``.
        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph API,
                              which provides a flexible API for aggressive fusion. Default to
                              ``False``, only valid when accelerator='jit', otherwise will
                              be ignored. For more details, please refer https://github.com/
                              pytorch/pytorch/tree/master/torch/csrc/jit/codegen/
                              onednn#pytorch---onednn-graph-api-bridge.
        :param q_config: Qconfig (https://pytorch.org/docs/stable/generated/torch.quantization.
                         qconfig.QConfig.html#qconfig) describes how to quantize a layer or a part
                         of the network by providing settings (observer classes) for activations
                         and weights respectively. Note that QConfig needs to contain observer
                         classes (like MinMaxObserver) or a callable that returns instances on
                         invocation, not the concrete observer instances themselves. Quantization
                         preparation function will instantiate observers multiple times for each
                         of the layers.
                         This parameter only works for native ipex and jit quantization with int8
                         precision. When accelerator='jit', we also support and recommend to pass
                         a QConfigMapping instead of single Qconfig for customized quantization.
                         QConfigMapping (https://pytorch.org/docs/stable/generated/torch.ao.
                         quantization.qconfig_mapping.QConfigMapping.html#qconfigmapping) is a
                         collection of quantization configurations, user can set the qconfig for
                         each operator (torch op calls, functional calls, module calls) in the
                         model through qconfig_mapping.
        :param output_tensors: boolean, default to True and output of the model will be Tensors,
                               only valid when accelerator='onnxruntime' or accelerator='openvino',
                               otherwise will be ignored. If output_tensors=False, output of the
                               export model will be ndarray.
        :param example_kwarg_inputs: a pack of keyword arguments of example inputs that will be
                                     passed to ``torch.jit.trace``. Default: ``None``. Either
                                     this argument or ``input_sample`` should be specified. The
                                     dict will be unpacking by the arguments name of the traced
                                     function. Only valid when accelerator='jit' and torch>=2.0,
                                     otherwise will be ignored.
        :param **kwargs: Other extra advanced settings include:
                         1. those be passed to ``torch.onnx.export`` function,
                         only valid when accelerator='onnxruntime'/'openvino',
                         otherwise will be ignored.
                         Possible arguments are: input_names, output_names, opset_version,
                         et al. For more details, please refer
                         https://pytorch.org/docs/stable/onnx.html#torch.onnx.export.
                         2. those be passed to ``model optimizer`` function of openvino,
                         only valid when accelerator='openvino',
                         otherwise will be ignored.
                         Possible arguments are: mean_values, layout, input, output, et al.
                         For more details about model optimizer, you can see mo --help .
                         If you want to quantize with openvino on VPUX device,
                         you must specify  ``mean_value`` for model optimizer function.
                         Here ``mean_value`` represents mean values to be used for the input image
                         per channel. Values to be provided in the (R,G,B) or [R,G,B] format.
                         Can be defined for desired input of the model, for example:
                         "--mean_values data[255,255,255],info[255,255,255]". The exact meaning
                         and order of channels depend on how the original model was trained.
        :return:            A accelerated torch.nn.Module if quantization is successful.
        """
        invalidInputError(precision in ['int8', 'fp16', 'bf16'],
                          "Only support 'int8', 'bf16', 'fp16' now, "
                          "no support for {}.".format(precision))
        # device name might be: CPU, GPU, GPU.0, VPUX ...
        invalidInputError(device == 'CPU' or 'GPU' in device or device == 'VPUX',
                          "Now we only support CPU, GPU and VPUX, not {}".format(device))
        if device != 'CPU' and accelerator not in ['openvino', None]:
            invalidInputError(False,
                              "Now we only support {} device when accelerator"
                              "is openvino.".format(device))
        model.eval()  # change model to eval mode
        if precision == 'bf16':
            if accelerator is None or accelerator == "jit":
                if use_ipex or accelerator == "jit":
                    if use_ipex is True:
                        invalidInputError(not TORCH_VERSION_LESS_1_10,
                                          "torch version should >=1.10 to use ipex")
                    use_jit = (accelerator == "jit")
                    if use_jit:
                        invalidInputError(jit_method in [None, 'trace', 'script'],
                                          "jit_method {} is invalid.".format(jit_method))
                    return PytorchIPEXJITBF16Model(model, input_sample=input_sample,
                                                   use_ipex=use_ipex, use_jit=use_jit,
                                                   channels_last=channels_last,
                                                   thread_num=thread_num, inplace=inplace,
                                                   jit_strict=jit_strict,
                                                   jit_method=jit_method,
                                                   weights_prepack=weights_prepack,
                                                   enable_onednn=enable_onednn,
                                                   example_kwarg_inputs=example_kwarg_inputs)
                else:
                    bf16_model = BF16Model(model, channels_last=channels_last,
                                           input_sample=input_sample,
                                           thread_num=thread_num)
                    return bf16_model
            elif accelerator == "openvino":
                invalidInputError(device == 'CPU',
                                  "Device {} don't support bfloat16.".format(device))
                final_openvino_option = {"INFERENCE_PRECISION_HINT": "bf16"}
                if openvino_config is not None:
                    final_openvino_option.update(openvino_config)
                return PytorchOpenVINOModel(model, input_sample,
                                            thread_num=thread_num,
                                            device=device,
                                            dynamic_axes=dynamic_axes,
                                            logging=logging,
                                            config=final_openvino_option,
                                            output_tensors=output_tensors,
                                            **kwargs)
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid for BF16.".format(accelerator))
        if precision == 'int8':
            # transform non-dataloader to dataloader
            if calib_data is not None and not isinstance(calib_data, DataLoader):
                dataset = RepeatDataset(sample=calib_data, num=1)
                calib_dataloader = DataLoader(dataset, batch_size=1)
                calib_dataloader = remove_batch_dim_fn(calib_dataloader)
            else:
                if calib_data is None and calib_dataloader is not None:
                    # will be deprecated in future release
                    warnings.warn("`calib_dataloader` will be deprecated in future release, please"
                                  "use `calib_data` instead.",
                                  category=DeprecationWarning)
                    calib_dataloader = calib_dataloader
                else:
                    calib_dataloader = calib_data
            # judge whether contains label in calib_dataloader
            # if not, will append label at last
            if accelerator is not None:
                calib_dataloader = automatic_add_label_in_dataloader(model,
                                                                     calib_dataloader,
                                                                     input_sample)

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
                        model = InferenceOptimizer.trace(
                            model,
                            input_sample=input_sample,
                            accelerator='onnxruntime',
                            thread_num=thread_num,
                            onnxruntime_session_options=onnxruntime_session_options,
                            simplification=simplification,
                            dynamic_axes=dynamic_axes,
                            output_tensors=output_tensors,
                            **kwargs)
                """
                If accelerator==None, quantized model returned should be an object of PytorchModel
                which is defined by neural-compressor containing a `GraphModule` for inference.
                Otherwise accelerator=='onnxruntime', it returns an ONNXModel object. A supported
                model which is able to run on Pytorch or ONNXRuntime can be fetched by
                `quantized_model.model`.
                """
                inc_quantize_arguments = {"model": model,
                                          "dataloader": inc_calib_dataloader,
                                          "eval_func": eval_func,
                                          "metric": metric, "thread_num": thread_num,
                                          "framework": framework, "conf": conf,
                                          "approach": approach,
                                          "tuning_strategy": tuning_strategy,
                                          "accuracy_criterion": accuracy_criterion,
                                          "timeout": timeout,
                                          "max_trials": max_trials,
                                          "onnxruntime_session_options": onnxruntime_session_options
                                          }
                if accelerator == "onnxruntime":
                    q_model = inc_quantize(**inc_quantize_arguments)
                    q_model.output_tensors = output_tensors
                    return q_model
                elif framework != 'pytorch_ipex':
                    return inc_quantize(**inc_quantize_arguments)
                else:
                    try:
                        return inc_quantize(**inc_quantize_arguments)
                    except Exception:
                        # use pure ipex quantization as a backup for inc ipex quantization
                        return PytorchIPEXQuantizationModel(model,
                                                            inc_calib_dataloader,
                                                            q_config=q_config,
                                                            input_sample=input_sample,
                                                            channels_last=channels_last,
                                                            thread_num=thread_num,
                                                            inplace=inplace,
                                                            jit_strict=jit_strict,
                                                            enable_onednn=enable_onednn)
            elif accelerator == 'openvino':
                model_type = type(model).__name__
                if not model_type == 'PytorchOpenVINOModel':
                    if input_sample is None:
                        # input_sample can be a dataloader
                        input_sample = calib_dataloader
                    # For CPU: fp32 -> int8, for GPU/VPUX: fp16 -> int8
                    _precision = 'fp16' if device != 'CPU' else 'fp32'
                    if device == 'VPUX':
                        # for fp16 on VPUX, must specify mean_value.
                        invalidInputError('mean_value' in kwargs,
                                          "If you want to quantize with openvino on VPUX device, "
                                          "you must specify mean_value for model optimizer "
                                          "function. For more details about model optimizer, you "
                                          "can see mo --help .")
                    model = PytorchOpenVINOModel(model, input_sample,
                                                 precision=_precision,
                                                 thread_num=thread_num,
                                                 device=device,
                                                 dynamic_axes=dynamic_axes,
                                                 logging=logging,
                                                 config=openvino_config,
                                                 output_tensors=output_tensors,
                                                 **kwargs)
                invalidInputError(type(model).__name__ == 'PytorchOpenVINOModel',
                                  "Invalid model to quantize. Please use a nn.Module or a model "
                                  "from InferenceOptimizer.trace(accelerator=='openvino')")
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
            elif accelerator == 'jit':
                invalidInputError(jit_method in [None, 'trace', 'script'],
                                  "jit_method {} is invalid.".format(jit_method))
                return PytorchJITINT8Model(model,
                                           calib_dataloader,
                                           q_config=q_config,
                                           input_sample=input_sample,
                                           channels_last=channels_last,
                                           thread_num=thread_num,
                                           jit_strict=jit_strict,
                                           jit_method=jit_method,
                                           enable_onednn=enable_onednn,
                                           example_kwarg_inputs=example_kwarg_inputs)
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid.".format(accelerator))
        if precision == 'fp16':
            invalidInputError(accelerator in ['openvino', None],
                              "fp16 is not supported on {} accelerator.".format(accelerator))
            if device == 'VPUX':
                # for fp16 on VPUX, must specify mean_value.
                invalidInputError('mean_value' in kwargs,
                                  "If you want to quantize with openvino float16 precision on "
                                  "VPUX device, you must specify mean_value for model optimizer "
                                  "function. For more details about model optimizer, you can "
                                  "see mo --help .")
            if accelerator is None and device == "GPU":
                return PytorchIPEXPUModel(model, thread_num=thread_num,
                                          precision=precision, use_ipex=use_ipex).to("xpu")
            return PytorchOpenVINOModel(model, input_sample,
                                        precision=precision,
                                        thread_num=thread_num,
                                        device=device,
                                        dynamic_axes=dynamic_axes,
                                        logging=logging,
                                        config=openvino_config,
                                        output_tensors=output_tensors,
                                        **kwargs)

        invalidInputError(False,
                          "Precision {} is invalid.".format(precision))

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator: Optional[str] = None,
              use_ipex: bool = False,
              channels_last: bool = False,
              thread_num: Optional[int] = None,
              device: Optional[str] = 'CPU',
              onnxruntime_session_options=None,
              openvino_config=None,
              simplification: bool = True,
              jit_strict: bool = True,
              jit_method: Optional[str] = None,
              dynamic_axes: Union[bool, dict] = True,
              logging: bool = True,
              inplace: bool = False,
              weights_prepack: Optional[bool] = None,
              enable_onednn: bool = False,
              output_tensors: bool = True,
              strict_check: bool = True,
              example_kwarg_inputs=None,
              **kwargs):
        """
        Trace a torch.nn.Module and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param model: A torch.nn.Module model, including pl.LightningModule.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch
                            backend. 'openvino', 'onnxruntime' and 'jit' are supported for now.
        :param use_ipex: Whether we use ipex as accelerator for inferencing. Only valid when
                         accelerator='jit'/None, otherwise will be ignored. Default: ``False``.
        :param channels_last: Whether use channels last memory format, i.e. NHWC (batch size,
                              height, width, channels), as an alternative way to store tensors in
                              classic/contiguous NCHW order. This setting only works for 4-dim
                              Tensor. Default: ``False``.
        :param thread_num: (optional) An int represents how many threads(cores) is needed for
                           inference. This parameter only controls the usage of thread number in
                           later inference process of your obtained accelerated model. In other
                           words, the process of model conversion won't be restricted by this
                           parameter.
        :param device: (optional) A string represents the device of the inference. Default to 'CPU',
                                  vaild choices are 'CPU'/'GPU'. 'GPU' is only valid when
                                  accelerator="openvino"/None. IPEX will be forcely used if
                                  accelerator=None.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param simplification: Whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
        :param jit_strict: Whether recording your mutable container types. This parameter will be
                           passed to ``torch.jit.trace``. if ``accelerator != 'jit'`` or
                           ``jit_method='script'``, it will be ignored. Default to True.
        :param jit_method: Whether to use ``jit.trace`` or ``jit.script`` to convert a model
                           to TorchScript. Accepected values are ``'trace'``, ``'script'``,
                           and ``None``. Default to be ``None`` meaning the try-except logic
                           to use ``jit.trace`` or ``jit.script``. If ``accelerator != 'jit'``,
                           this parameter will be ignored.
        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model
                             will have the first dim of each Tensor input as a dynamic batch_size.
                             If dynamic_axes=False, the exported model will have the shapes of all
                             input and output tensors set to exactly match those given in
                             input_sample. To specify axes of tensors as dynamic (i.e. known only
                             at run-time), set dynamic_axes to a dict with schema:

                             | KEY (str): an input or output name. Each name must also be provided
                             | in input_names or output_names.
                             |
                             | VALUE (dict or list): If a dict, keys are axis indices and values
                             | are axis names. If a list, each element is an axis index.

                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.
        :param logging: Whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :param inplace: whether to perform inplace optimization. Default: ``False``.
        :param weights_prepack: Whether to perform weight prepack for convolution and linear to
                                avoid oneDNN weights reorder. The default value is None. Explicitly
                                setting this knob overwrites the configuration set by level knob.
                                Only valid when ``use_ipex=True``, otherwise will be ignored.
                                You can try to reduce the occupied memory size by setting this
                                parameter to ``False``.
        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph API,
                              which provides a flexible API for aggressive fusion. Default to
                              ``False``, only valid when accelerator='jit', otherwise will be
                              ignored. For more details, please refer https://github.com/pytorch/
                              pytorch/tree/master/torch/csrc/jit/codegen/
                              onednn#pytorch---onednn-graph-api-bridge.
        :param output_tensors: boolean, default to ``True`` and output of the model will be Tensors,
                               only valid when accelerator='onnxruntime' or accelerator='openvino',
                               otherwise will be ignored. If output_tensors=False, output of the
                               export model will be ndarray.
        :param strict_check: some checking in ``trace`` is non-trivial while not critical for the
                             optimization (e.g., if the model is a nn.Module or its subclass). This
                             param helps to eliminate the not critical checking, which may enable
                             more models to be optimized while may bring some strange error
                             message. Default to ``True``.
        :param example_kwarg_inputs: a pack of keyword arguments of example inputs that will be
                                     passed to ``torch.jit.trace``. Default: ``None``. Either
                                     this argument or ``input_sample`` should be specified. The
                                     dict will be unpacking by the arguments name of the traced
                                     function. Only valid when accelerator='jit' and torch>=2.0,
                                     otherwise will be ignored.
        :param **kwargs: Other extra advanced settings include:
                         1. those be passed to torch.onnx.export function,
                         only valid when accelerator='onnxruntime'/'openvino',
                         otherwise will be ignored.
                         Possible arguments are: input_names, output_names, opset_version,
                         et al. For more details, please refer
                         https://pytorch.org/docs/stable/onnx.html#torch.onnx.export.
                         2. those be passed to model optimizer function of openvino,
                         only valid when accelerator='openvino',
                         otherwise will be ignored.
                         Possible arguments are: mean_values, layout, input, output, et al.
                         For more details about model optimizer, you can see mo --help .
        :return: Model with different acceleration.
        """
        if strict_check:
            invalidInputError(
                isinstance(model, nn.Module) and not isinstance(model, AcceleratedLightningModule),
                "Expect a nn.Module instance that is not traced or quantized"
                "but got type {}".format(type(model))
            )
        # device name might be: CPU, GPU, GPU.0 ...
        invalidInputError(device == 'CPU' or 'GPU' in device,
                          "Now we only support fp32 for CPU and GPU, not {}".format(device))
        # can't set precision for trace
        invalidInputError("precision" not in kwargs,
                          "Don't pass precision when call InferenceOptimizer.trace, otherwise you "
                          "should call InferenceOptimizer.quantize(precision=...)")
        if device != 'CPU' and accelerator not in ('openvino', None):
            invalidInputError(False,
                              "Now we only support {} device when accelerator "
                              "is openvino or None.".format(device))
        try:
            model.eval()  # change model to eval mode
        except Exception:
            pass
        if accelerator == 'openvino':  # openvino backend will not care about ipex usage
            final_openvino_option = {"INFERENCE_PRECISION_HINT": "f32"}
            if openvino_config is not None:
                final_openvino_option.update(openvino_config)
            return PytorchOpenVINOModel(model, input_sample,
                                        thread_num=thread_num,
                                        device=device,
                                        dynamic_axes=dynamic_axes,
                                        logging=logging,
                                        config=final_openvino_option,
                                        output_tensors=output_tensors,
                                        **kwargs)
        if accelerator == 'onnxruntime':  # onnxruntime backend will not care about ipex usage
            if onnxruntime_session_options is None:
                import onnxruntime
                onnxruntime_session_options = onnxruntime.SessionOptions()
            if thread_num is not None:
                onnxruntime_session_options.intra_op_num_threads = thread_num
                onnxruntime_session_options.inter_op_num_threads = thread_num
            return PytorchONNXRuntimeModel(model, input_sample,
                                           onnxruntime_session_options,
                                           simplification=simplification,
                                           dynamic_axes=dynamic_axes,
                                           output_tensors=output_tensors,
                                           **kwargs)
        if accelerator is None and device == "GPU":
            return PytorchIPEXPUModel(model, thread_num=thread_num, use_ipex=use_ipex).to("xpu")
        if (accelerator == 'jit' or use_ipex is True or channels_last is True) and device == "CPU":
            if use_ipex:
                invalidInputError(not TORCH_VERSION_LESS_1_10,
                                  "torch version should >= 1.10 to use ipex")
            use_jit = (accelerator == "jit")
            if use_jit:
                invalidInputError(jit_method in [None, 'trace', 'script'],
                                  "jit_method {} is invalid.".format(jit_method))
            return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex,
                                       use_jit=use_jit, channels_last=channels_last,
                                       thread_num=thread_num, inplace=inplace,
                                       jit_strict=jit_strict, jit_method=jit_method,
                                       weights_prepack=weights_prepack,
                                       enable_onednn=enable_onednn,
                                       example_kwarg_inputs=example_kwarg_inputs)
        invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))

    @staticmethod
    def get_context(model: nn.Module, *models):
        """
        Obtain corresponding context manager from (multi) model, defaults to BaseContextManager().

        :param model: Any model of torch.nn.Module, including all models accelareted by
               InferenceOptimizer.trace/InferenceOptimizer.quantize.
        :param models: Any model of torch.nn.Module or list of torch.nn.Module, including all models
               accelareted by InferenceOptimizer.trace/InferenceOptimizer.quantize.
        :return: a context manager if there is no conflict between context managers,
                 otherwise will report RuntimeError.
        """
        def obtain_manager(input_model):
            if hasattr(input_model, "_nano_context_manager"):
                _context_manager = input_model._nano_context_manager
            else:
                _context_manager = generate_context_manager(accelerator=None,
                                                            precision="fp32")
            return _context_manager

        def join_manager(manager1, manager2):
            def is_bf16(x):
                return isinstance(x, AutocastContextManager)
            if (is_bf16(manager1) ^ is_bf16(manager2)) is True:
                warnings.warn("Only one of the context managers uses mixed precision, "
                              "and we will return the context manager with mixed precision.")
            manager = None
            if is_bf16(manager1) or is_bf16(manager2):
                manager = AutocastContextManager()
            else:
                manager = BaseContextManager()
            thread_num1 = manager1.thread_num
            thread_num2 = manager2.thread_num
            if thread_num1 != thread_num2:
                if thread_num1 is None or thread_num2 is None:
                    warnings.warn("One of the two models has thread control and the other "
                                  "does not. The returned context manager will be dominated "
                                  "by the non-None one.")
                else:
                    warnings.warn("These context managers have different thread_num.  We will "
                                  "set thread_num to the larger one.")
            if thread_num1 is None:
                manager.thread_num = thread_num2
            elif thread_num2 is None:
                manager.thread_num = thread_num1
            else:
                manager.thread_num = max(thread_num1, thread_num2)
            return manager

        _context_manager = obtain_manager(model)
        if len(models) == 0:
            # only single model
            return _context_manager
        else:
            for model_ in models:
                if isinstance(model_, nn.Module):
                    new_manager = obtain_manager(model_)
                    _context_manager = join_manager(_context_manager, new_manager)
            return _context_manager

    @staticmethod
    def save(model: nn.Module, path, compression="fp32"):
        """
        Save the model to local file.

        :param model: Any model of torch.nn.Module, including all models accelareted by
               InferenceOptimizer.trace/InferenceOptimizer.quantize.
        :param path: Path to saved model. Path should be a directory.
        :param compression: str. This parameter only effective for jit, ipex or pure
               pytorch model with fp32 or bf16 precision. Defaultly, all models are saved
               by dtype=fp32 for their parameters. If users set a lower precision, a smaller
               file sill be saved with some accuracy loss. Users always need to use nano
               to load the compressed file if compression is set other than "fp32".
               Currently, "bf16" and "fp32"(default) are supported.
        """
        save_model(model, path, compression)

    @staticmethod
    def load(path, model: Optional[nn.Module] = None, input_sample=None,
             inplace=False, device=None, cache_dir=None, shapes=None):
        """
        Load a model from local.

        :param path: Path to model to be loaded. Path should be a directory.
        :param model: Required FP32 model to load pytorch model, it is needed if:
               1. you accelerate the model with accelerator=None by
               InferenceOptimizer.trace()/InferenceOptimizer.quantize().
               2. you accelerate the model with InferenceOptimizer.optimize() and
               get_model()/get_best_model(), and the best method or the method you
               specify don't contain accelerator 'onnxruntime'/'openvino'/'jit'.
               If you are not sure what optimization method is used, we recommend that
               you always pass in the original model for this case.
               3. you want to the loaded model contains the attributes of original model.
        :param input_sample: Input sample for your model, could be a Tensor or a tuple.
               This parameter is needed if:
               1. saving model is accelerated by INC IPEX quantization.
               2. saving model is accelerated by JIT and you set compression='bf16'
               when saving.
        :param inplace: whether to perform inplace optimization. Default: ``False``.
        :param device: A string represents the device of the inference. Default to None.
               Only valid for openvino model, otherwise will be ignored.
        :param cache_dir: A directory for OpenVINO to cache the model. Default to None.
               Only valid for openvino model, otherwise will be ignored.
        :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',
               '[1,3,224,224]'. This parameter affect model Parameter shape, can be
               dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.
               Default to None, which means you don't want to reshape the model inputs.
               Only valid for openvino model, otherwise will be ignored.
        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or
                 precision(FP32/FP16/BF16/INT8).
        """
        return load_model(path, model, input_sample=input_sample,
                          inplace=inplace, device=device, cache_dir=cache_dir,
                          shapes=shapes)

    @staticmethod
    def to_multi_instance(model: nn.Module, num_processes: int = 4,
                          cores_per_process: int = None,
                          cpu_for_each_process: List[List[int]] = None) -> _MultiInstanceModel:
        """
        Transform a model to multi-instance inference model.

        :param model: The model to transform.
        :param num_processes: The number of processes to use, default to 4.
        :param cores_per_process: Number of CPU cores used by each process,
            default to `None`, means decided automatically.
        :param cpu_for_each_process: Specify the CPU cores used by each process,
            default to `None`, if set, it will override `num_processes` and `cores_per_process`.
        :return: Model with multi-instance inference acceleration.
        """
        invalidInputError(isinstance(num_processes, int) and num_processes > 0,
                          "num_processes must be a positive integer")
        # if num_processes == 1 and no core specification, we will use current process directly
        if num_processes == 1 and cpu_for_each_process is None and cores_per_process is None:
            warnings.warn("Will run inference in current process directly "
                          "because the `num_processes` is 1")
            return _MultiInstanceModel(model, None, None, None, None)

        p_num = num_processes if cpu_for_each_process is None else len(cpu_for_each_process)

        # else we will start multiple sub-processes
        send_queue = mp.Queue()
        recv_queue = mp.Queue()
        next_idx = mp.Value('i', 0, lock=True)

        if cpu_for_each_process is None:
            if cores_per_process is None:
                envs = schedule_processors(p_num)
            else:
                envs = [{
                    "KMP_AFFINITY": f"granularity=fine,proclist="
                                    f"[{i*cores_per_process}-{(i+1)*cores_per_process-1}]"
                                    f",explicit",
                    "OMP_NUM_THREADS": str(cores_per_process)
                } for i in range(p_num)]
        else:
            envs = [{
                "KMP_AFFINITY": f"granularity=fine,proclist="
                                f"[{','.join([str(i) for i in cpu_for_each_process[i]])}]"
                                f",explicit",
                "OMP_NUM_THREADS": str(len(cpu_for_each_process[i]))
            } for i in range(p_num)]
        ps = []
        for i in range(p_num):
            with EnvContext(envs[i]):
                p = mp.Process(target=_multi_instance_helper,
                               args=(model, send_queue, recv_queue, next_idx), daemon=True)
                p.start()
                ps.append(p)

        return _MultiInstanceModel(model, ps, send_queue, recv_queue, next_idx)


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


def _obtain_combinations(all_combinations, precision, accelerator, use_ipex):
    new_combinations = {}
    new_combinations["original"] = all_combinations["original"]
    for method, option in all_combinations.items():
        if precision is not None:
            if option.get_precision() not in precision:
                continue
        if accelerator is not None:
            if option.get_accelerator() not in accelerator:
                continue
        if use_ipex is not None:
            if option.ipex != use_ipex:
                continue
        new_combinations[method] = option
    return new_combinations
