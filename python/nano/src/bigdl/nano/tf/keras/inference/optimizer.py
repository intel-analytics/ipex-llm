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

import os
import subprocess
import tempfile
import cloudpickle
import copy
import time
import operator
from pathlib import Path
import numpy as np
import traceback
import inspect
import tensorflow as tf
import keras
from typing import Dict, Optional, List, Union, Callable
from bigdl.nano.utils.common import BaseInferenceOptimizer, available_acceleration_combination,\
    AccelerationOption, latency_calculate_helper, format_optimize_result
from bigdl.nano.tf.utils import patch_compiled_and_attrs, patch_attrs
from bigdl.nano.tf.utils import _ModuleWrapper
from bigdl.nano.utils.common import invalidInputError
from tensorflow.keras import Model as Model
from tensorflow.data import Dataset
from tensorflow.keras.metrics import Metric
from bigdl.nano.deps.neural_compressor.inc_api import quantize as inc_quantzie
from bigdl.nano.deps.openvino.openvino_api import KerasOpenVINOModel
from bigdl.nano.deps.onnxruntime.onnxruntime_api import KerasONNXRuntimeModel
from bigdl.nano.deps.openvino.openvino_api import load_openvino_model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import load_onnxruntime_model
from bigdl.nano.deps.neural_compressor.inc_api import load_inc_model
from bigdl.nano.tf.keras.amp import BF16Model, load_bf16_model
from bigdl.nano.utils.common import compare_version


class TFAccelerationOption(AccelerationOption):
    def optimize(self, model, x=None, y=None, input_spec=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        accelerator = self.get_accelerator()
        if self.get_precision() == "fp32":
            # trace
            if accelerator is None:
                return model
            else:
                acce_model = InferenceOptimizer.trace(model=model,
                                                      accelerator=accelerator,
                                                      input_spec=input_spec,
                                                      thread_num=thread_num,
                                                      # remove output of openvino
                                                      logging=logging)
        else:
            # quantize
            ort_method: str = self.method
            acce_model = InferenceOptimizer.quantize(model=model,
                                                     precision=self.get_precision(),
                                                     accelerator=accelerator,
                                                     input_spec=input_spec,
                                                     x=x,
                                                     y=y,
                                                     method=ort_method,
                                                     thread_num=thread_num,
                                                     sample_size=sample_size_for_pot,
                                                     # remove output of openvino
                                                     logging=logging)
        return acce_model


class InferenceOptimizer(BaseInferenceOptimizer):

    # acceleration method combinations, developers may want to register some new
    # combinations here
    ALL_INFERENCE_ACCELERATION_METHOD: Dict = \
        {  # type: ignore
            "original": TFAccelerationOption(),
            "static_int8": TFAccelerationOption(inc=True),
            "openvino_fp32": TFAccelerationOption(openvino=True),
            "openvino_int8": TFAccelerationOption(openvino=True, pot=True),
            "onnxruntime_fp32": TFAccelerationOption(onnxruntime=True),
            "onnxruntime_int8_qlinear": TFAccelerationOption(onnxruntime=True, inc=True,
                                                             method="qlinear"),
            "onnxruntime_int8_integer": TFAccelerationOption(onnxruntime=True, inc=True,
                                                             method="integer"),
        }  # type: ignore

    def optimize(self, model: Model,
                 x: Union[tf.Tensor, np.ndarray, tf.data.Dataset],
                 y: Union[tf.Tensor, np.ndarray] = None,
                 validation_data: Optional[Dataset] = None,
                 input_spec=None,
                 batch_size: int = 1,
                 metric: Optional[Metric] = None,
                 direction: str = "max",
                 thread_num: Optional[int] = None,
                 logging: bool = False,
                 latency_sample_num: int = 100,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None,
                 output_filename: Optional[str] = None) -> None:
        '''
        This function will give all available inference acceleration methods a try
        and record the latency, accuracy and model instance inside the Optimizer for
        future usage. All model instance is setting to eval mode.

        The available methods are "original", "openvino_fp32", "onnxruntime_fp32", "int8".

        :param model: A keras.Model to be optimized
        :param x: Input data which is used for training. It could be:

                  | 1. a Numpy array (or array-like), or a list of arrays (in case the model
                  | has multiple inputs).
                  |
                  | 2. a TensorFlow tensor, or a list of tensors (in case the model has
                  | multiple inputs).
                  |
                  | 3. an unbatched tf.data.Dataset. Should return a tuple of (inputs, targets).

                  X will be used as calibration dataset for Post-Training Static Quantization (PTQ),
                  as well as be used for generating input_sample to calculate latency.
                  To avoid data leak during calibration, please use training dataset.
        :param y: Target data. Like the input data x, it could be either Numpy array(s) or
                  TensorFlow tensor(s). Its length should be consistent with x.
                  If x is a dataset, y will be ignored (since targets will be obtained from x).
        :param validation_data: (optional) An unbatched tf.data.Dataset object for accuracy
               evaluation. This is only needed when users care about the possible accuracy drop.
        :param input_spec: (optional) A (tuple or list of) ``tf.TensorSpec``
                           defining the shape/dtype of the input. This is only required when
                           you have a custom Keras model (no input/output layer is explicitly
                           defined).
        :param metric: (optional) A tensorflow.keras.metrics.Metric object which is used for
               calculating accuracy.
        :param direction: (optional) A string that indicates the higher/lower
               better for the metric, "min" for the lower the better and "max" for the
               higher the better. Default value is "max".
        :param thread_num: (optional) An int represents how many threads(cores) is needed for
               inference. This parameter only controls the usage of thread number in the process
               of latency calculation as well as later inference process of your obtained
               accelerated model. In other words, the process of model conversion and optional
               accuracy calculation won't be restricted by this parameter. Defaults to None,
               represents that all cores will be used.
        :param logging: whether to log detailed information of model conversion.
               Default: False.
        :param latency_sample_num: (optional) a int represents the number of repetitions
               to calculate the average latency. The default value is 100.
        :param includes: (optional) a list of acceleration methods that will be included in the
               search. Default to None meaning including all available methods. "original" method
               will be automatically add to includes.
        :param excludes: (optional) a list of acceleration methods that will be excluded from the
               search. "original" will be ignored in the excludes.
        :param output_filename: (optional) a string filename is used to specify the file which the
               optimized table will be writed. The default is None which means don't write to file.
        '''
        # check if model is a nn.Module or inherited from a nn.Module
        invalidInputError(isinstance(model, Model), "model should be a Keras Model.")
        invalidInputError(direction in ['min', 'max'],
                          "Only support direction 'min', 'max'.")

        # get the available methods whose dep is met
        available_dict: Dict =\
            available_acceleration_combination(excludes=excludes,
                                               includes=includes,
                                               full_methods=self.ALL_INFERENCE_ACCELERATION_METHOD)

        self._direction: str = direction  # save direction as attr
        # record whether calculate accuracy in optimize by this attr
        if validation_data is None or metric is None:
            self._calculate_accuracy = False
        else:
            # test whether accuracy calculation works later
            # make sure dataset don't have batch
            batched_validation_data = validation_data.batch(batch_size)
            self._calculate_accuracy = True

        if os.getenv('OMP_NUM_THREADS') is not None:
            default_threads: int = int(os.getenv('OMP_NUM_THREADS'))  # type: ignore
        else:
            default_threads = None  # type: ignore
        thread_num = default_threads if thread_num is None else int(thread_num)  # type: ignore

        result_map: Dict[str, Dict] = {}

        if isinstance(x, Dataset):
            batched_training_dataset = x.batch(batch_size)
            input_sample = next(iter(batched_training_dataset))
            if isinstance(input_sample, (list, tuple)) and len(input_sample) > 1:
                input_sample = input_sample[:-1]
        else:
            input_sample = tf.convert_to_tensor(x[:batch_size])

        if isinstance(input_sample, (list, tuple)) and len(input_sample) == 1:
            input_sample = input_sample[0]

        st = time.perf_counter()
        try:
            if isinstance(input_sample, tf.Tensor):
                model(input_sample)
            else:
                model(*input_sample)
        except Exception:
            invalidInputError(False,
                              "x is incompatible with your model input.")
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
                precision: str = option.get_precision()
                try:
                    acce_model = option.optimize(model=model,
                                                 x=x,
                                                 y=y,
                                                 input_spec=input_spec,
                                                 thread_num=thread_num,
                                                 logging=logging,
                                                 sample_size_for_pot=sample_size_for_pot)
                except Exception:
                    traceback.print_exc()
                    result_map[method]["status"] = "fail to convert"
                    print(f"----------Failed to convert to {method}----------")
                    continue

                result_map[method]["status"] = "successful"

                def func_test(model, sample):
                    model(sample)
                try:
                    if method in ("original", "static_int8") and thread_num is not None:
                        _flag = True  # represent whether subprocess works
                        # for original keras model, as tf.config.threading can't set thread
                        # during running, so here we use subprocess to calculate throughput
                        params = {"iterrun": latency_sample_num,
                                  "func": func_test,
                                  "model": model,  # save original model
                                  "input_sample": input_sample,
                                  "method": method}
                        with tempfile.TemporaryDirectory() as temp_dir:
                            if method != "original":
                                # save accelerated model
                                InferenceOptimizer.save(acce_model, temp_dir)
                            _filename = os.path.join(temp_dir, "params")
                            cloudpickle.dump(params, open(_filename, "wb"))
                            my_env = os.environ.copy()
                            my_env["OMP_NUM_THREADS"] = str(thread_num)
                            worker_file = os.path.join(
                                os.path.split(os.path.realpath(__file__))[0], "_worker.py")
                            try:
                                result = subprocess.run(["python", worker_file,
                                                         _filename, str(thread_num)],
                                                        capture_output=True,
                                                        universal_newlines=True,
                                                        env=my_env)
                                latency = float(result.stdout.strip())
                                result_map[method]["latency"] = latency
                            except Exception:
                                _flag = False
                    if method != "original" or thread_num is None or _flag is False:
                        result_map[method]["latency"], status =\
                            latency_calculate_helper(latency_sample_num, baseline_time,
                                                     func_test, acce_model, input_sample)
                        if status is False and method != "original":
                            result_map[method]["status"] = "early stopped"
                            continue
                except Exception:
                    traceback.print_exc()
                    result_map[method]["status"] = "fail to forward"
                    print(f"----------{method} failed to forward----------")
                    continue

                if self._calculate_accuracy:
                    # here we suppose trace don't change accuracy,
                    # so we jump it to reduce time cost of optimize
                    if precision == "fp32" and method != "original":
                        _accuracy = result_map["original"]["accuracy"]
                        _accuracy = round(_accuracy, 3)
                        result_map[method]["accuracy"] = str(_accuracy) + '*'
                    else:
                        if method == "original":
                            # test whether metric works
                            try:
                                result_map[method]["accuracy"] =\
                                    _accuracy_calculate_helper(acce_model, metric,
                                                               batched_validation_data)
                            except Exception:
                                traceback.print_exc()
                                self._calculate_accuracy = False
                        else:
                            result_map[method]["accuracy"] =\
                                _accuracy_calculate_helper(acce_model, metric,
                                                           batched_validation_data)
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
            # only show this line when there is accuracy data
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
    def trace(model: Model,
              accelerator: Optional[str] = None,
              input_spec=None,
              thread_num: Optional[int] = None,
              device: Optional[str] = 'CPU',
              onnxruntime_session_options=None,
              openvino_config=None,
              logging=True,
              **kwargs):
        """
        Trace a Keras model and convert it into an accelerated module for inference.

        :param model: The Keras model to trace.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Keras
                            backend. 'openvino' and 'onnxruntime' are supported for now.
        :param input_spec: (optional) A (tuple or list of) ``tf.TensorSpec``
                           defining the shape/dtype of the input. This is only required when
                           you have a custom Keras model (no input/output layer is explicitly
                           defined).
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param device: (optional) A string represents the device of the inference. Default to 'CPU',
                        only valid when accelerator='openvino', otherwise will be ignored.
                        'CPU', 'GPU' are supported for now.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :param **kwargs: Other extra advanced settings include those be passed to model optimizer
                         function of openvino, only valid when accelerator='openvino',
                         otherwise will be ignored.
                         Possible arguments are: mean_values, layout, input, output, et al.
                         For more details about model optimizer, you can see mo --help .
        :return: Model with different acceleration(OpenVINO/ONNX Runtime).
        """
        # device name might be: CPU, GPU, GPU.0, VPUX ...
        invalidInputError(device == 'CPU' or 'GPU' in device,
                          "Now we only support fp32 for CPU and GPU, not {}".format(device))
        if device != 'CPU' and accelerator != 'openvino':
            invalidInputError(False,
                              "Now we only support {} device when accelerator "
                              "is openvino.".format(device))
        if accelerator == 'openvino':
            final_openvino_option = {"INFERENCE_PRECISION_HINT": "f32"} if device is 'CPU' else {}
            if openvino_config is not None:
                final_openvino_option.update(openvino_config)
            result = KerasOpenVINOModel(model,
                                        input_spec=input_spec,
                                        precision='fp32',
                                        thread_num=thread_num,
                                        device=device,
                                        config=final_openvino_option,
                                        logging=logging,
                                        **kwargs)
        elif accelerator == 'onnxruntime':
            if onnxruntime_session_options is None:
                import onnxruntime
                onnxruntime_session_options = onnxruntime.SessionOptions()
            if thread_num is not None:
                onnxruntime_session_options.intra_op_num_threads = thread_num
                onnxruntime_session_options.inter_op_num_threads = thread_num
            result = KerasONNXRuntimeModel(model, input_spec, onnxruntime_session_options)
        else:
            invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))
        return patch_compiled_and_attrs(result, model)

    @staticmethod
    def quantize(model: Model,
                 x: Union[tf.Tensor, np.ndarray, tf.data.Dataset] = None,
                 y: Union[tf.Tensor, np.ndarray] = None,
                 precision: str = 'int8',
                 accelerator: Optional[str] = None,
                 input_spec=None,
                 eval_func: Optional[Callable] = None,
                 metric: Optional[Metric] = None,
                 accuracy_criterion: Optional[dict] = None,
                 approach: str = 'static',
                 method: Optional[str] = None,
                 conf: Optional[str] = None,
                 tuning_strategy: Optional[str] = None,
                 timeout: Optional[int] = None,
                 max_trials: Optional[int] = None,
                 batch: Optional[int] = None,
                 thread_num: Optional[int] = None,
                 device: Optional[str] = 'CPU',
                 inputs: List[str] = None,
                 outputs: List[str] = None,
                 sample_size: int = 100,
                 onnxruntime_session_options=None,
                 openvino_config=None,
                 logging: bool = True,
                 **kwargs):
        """
        Post-training quantization on a keras model.

        :param model: The Keras model to quantize.
        :param x: Input data which is used for training. It could be:

                  | 1. a Numpy array (or array-like), or a list of arrays (in case the model
                  | has multiple inputs).
                  |
                  | 2. a TensorFlow tensor, or a list of tensors (in case the model has
                  | multiple inputs).
                  |
                  | 3. an unbatched tf.data.Dataset. Should return a tuple of (inputs, targets).

                  X will be used as calibration dataset for Post-Training Static Quantization (PTQ).
                  To avoid data leak during calibration, please use training dataset.
                  only valid when precision='int8', otherwise will be ignored.
        :param y: Target data. Like the input data x, it could be either Numpy array(s) or
                  TensorFlow tensor(s). Its length should be consistent with x.
                  If x is a dataset, y will be ignored (since targets will be obtained from x).
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', 'bf16', 'fp16', defaults to 'int8'.
                                Note that, mixed bf16 precision only works for ``keras.Model`` with
                                explict input and output definition(e.g.,
                                model = keras.Model(inputs=inputs, outputs=outputs)).
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', defaults to None.
                                None means staying in tensorflow.
        :param input_spec:      (optional) A (tuple or list of) ``tf.TensorSpec``
                                defining the shape/dtype of the input. This is only required when
                                you have a custom Keras model (no input/output layer is explicitly
                                defined).
        :param eval_func:       A evaluation function which only accepts model as input and return
                                evaluation value. This parameter provides a higher degree of
                                freedom than using eval_loader and metric. Default to None meaning
                                no performance tuning, but it would be better give an evaluation
                                function to get better quantization performance.
        :param metric:          A tensorflow.keras.metrics.Metric object for evaluation.
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
        :param approach:        'static' or 'dynamic'.
                                'static': post_training_static_quant,
                                'dynamic': post_training_dynamic_quant.
                                Default: 'static'. Only 'static' approach is supported now.
        :param method:      Method to do quantization. When accelerator=None, supported methods:
                None. When accelerator='onnxruntime', supported methods: 'qlinear', 'integer',
                defaults to 'qlinear'. Suggest 'qlinear' for lower accuracy drop if using
                static quantization.
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
        :param batch:       Batch size of dataloader for calib_dataset. Defaults to None, if the
                            dataset is not a BatchDataset, batchsize equals to 1. Otherwise,
                            batchsize complies with the dataset._batch_size.
        :param thread_num:  (optional) a int represents how many threads(cores) is needed for
                            inference, only valid for accelerator='onnxruntime'
                            or accelerator='openvino'.
        :param device: (optional) A string represents the device of the inference. Default to 'CPU',
                        only valid when accelerator='openvino', otherwise will be ignored.
                        'CPU', 'GPU' and 'VPUX' are supported for now.
        :param inputs:      A list of input names.
                            Default: None, automatically get names from graph.
        :param outputs:     A list of output names.
                            Default: None, automatically get names from graph.
        :param sample_size: (optional) a int represents how many samples will be used for
                            Post-training Optimization Tools (POT) from OpenVINO toolkit,
                            only valid for accelerator='openvino'. Default to 100.
                            The larger the value, the more accurate the conversion,
                            the lower the performance degradation, but the longer the time.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
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
        :return:            A TensorflowBaseModel. If there is no model found, return None.
        """
        invalidInputError(precision in ['int8', 'fp16', 'bf16'],
                          "Only support 'int8', 'bf16', 'fp16' now, "
                          "no support for {}.".format(precision))
        # device name might be: CPU, GPU, GPU.0, VPUX ...
        invalidInputError(device == 'CPU' or 'GPU' in device or device == 'VPUX',
                          "Now we only support CPU, GPU and VPUX, not {}".format(device))
        if device != 'CPU' and accelerator != 'openvino':
            invalidInputError(False,
                              "Now we only support {} device when accelerator "
                              "is openvino.".format(device))

        if isinstance(model, _ModuleWrapper):
            original_model = model.source_obj
            model = model.target_obj
        else:
            original_model = model

        if precision == 'fp16':
            invalidInputError(accelerator == 'openvino',
                              "fp16 is not supported on {} accelerator.".format(accelerator))
            if device == 'VPUX':
                # for fp16 on VPUX, must specify mean_value.
                invalidInputError('mean_value' in kwargs,
                                  "If you want to quantize with openvino float16 precision on "
                                  "VPUX device, you must specify mean_value for model optimizer "
                                  "function. For more details about model optimizer, you can "
                                  "see mo --help .")
            from bigdl.nano.deps.openvino.tf.model import KerasOpenVINOModel    # type: ignore
            result = KerasOpenVINOModel(model,
                                        input_spec=input_spec,
                                        precision=precision,
                                        thread_num=thread_num,
                                        device=device,
                                        config=openvino_config,
                                        logging=logging,
                                        **kwargs)
            return patch_compiled_and_attrs(result, original_model)

        elif precision == 'bf16':
            invalidInputError(accelerator == 'openvino' or accelerator is None,
                              "Accelerator {} is invalid for BF16.".format(accelerator))
            invalidInputError(device == 'CPU',
                              "Device {} don't support bfloat16.".format(device))
            if accelerator == 'openvino':
                final_openvino_option = {"INFERENCE_PRECISION_HINT": "bf16"}
                if openvino_config is not None:
                    final_openvino_option.update(openvino_config)
                from bigdl.nano.deps.openvino.tf.model import KerasOpenVINOModel    # type: ignore
                result = KerasOpenVINOModel(model,
                                            input_spec=input_spec,
                                            precision=precision,
                                            thread_num=thread_num,
                                            device=device,
                                            config=final_openvino_option,
                                            logging=logging,
                                            **kwargs)
            elif accelerator is None:
                return BF16Model(model)
            return patch_compiled_and_attrs(result, original_model)

        invalidInputError(approach == 'static', "Only 'static' approach is supported now.")

        if not isinstance(x, tf.data.Dataset) and y is None:
            # fake label to make quantization work
            y = range(len(x))    # type: ignore
        if isinstance(x, tf.data.Dataset):
            batch_data = next(iter(x))
            if isinstance(batch_data, tf.Tensor) or \
                    isinstance(batch_data, tuple) and len(batch_data) == 1:
                # fake label to make quantization work
                y = range(len(x))    # type: ignore
                y = tf.data.Dataset.from_tensor_slices(y)
                x = tf.data.Dataset.zip((x, y))
        if accelerator is None:
            if isinstance(x, tf.data.Dataset):
                calib_dataset = x
            else:
                calib_dataset = tf.data.Dataset.from_tensor_slices((x, y))
            if batch:
                calib_dataset = calib_dataset.batch(batch)

            saved_model_input_spec_set = model._saved_model_inputs_spec is not None
            if not model.built and not saved_model_input_spec_set or \
                    not hasattr(model, 'output_shape'):
                invalidInputError(input_spec is not None,
                                  "`input_spec` cannot be None when passing unbuilt model.")
                # model cannot be saved either because the input shape is not available
                # or because the forward pass of the model is not defined
                if isinstance(input_spec, (tuple, list)):
                    input_shape = (i.shape for i in input_spec)
                else:
                    input_shape = input_spec.shape
                _output_shape = model.compute_output_shape(input_shape)
            else:
                _output_shape = model.output_shape
            if model.inputs is None or model.outputs is None:
                INC_LESS_14 = compare_version("neural_compressor", operator.lt, "1.14")
                # oly works for inc version >= 1.14
                if not INC_LESS_14:
                    # try to fake input and output for model
                    signature = inspect.signature(model.call)
                    input_names = []
                    for param in signature.parameters.values():
                        input_names.append(param.name)
                    if inputs is None:
                        inputs = input_names
                    if outputs is None:
                        outputs = "outputs"    # type: ignore

            result = inc_quantzie(model, dataloader=calib_dataset,
                                  eval_func=eval_func,
                                  metric=metric,
                                  framework='tensorflow',
                                  conf=conf,
                                  approach=approach,
                                  tuning_strategy=tuning_strategy,
                                  accuracy_criterion=accuracy_criterion,
                                  timeout=timeout,
                                  max_trials=max_trials,
                                  inputs=inputs,
                                  outputs=outputs)
            result._output_shape = _output_shape
        elif accelerator == 'openvino':
            from bigdl.nano.deps.openvino.tf.model import KerasOpenVINOModel    # type: ignore
            if isinstance(model, KerasOpenVINOModel):    # type: ignore
                openvino_model = model
            else:
                # For CPU: fp32 -> int8, for GPU: fp16 -> int8
                _precision = 'fp16' if device != 'CPU' else 'fp32'
                if device == 'VPUX':
                    # for fp16 on VPUX, must specify mean_value.
                    invalidInputError('mean_value' in kwargs,
                                      "If you want to quantize with openvino on VPUX device, "
                                      "you must specify mean_value for model optimizer "
                                      "function. For more details about model optimizer, you "
                                      "can see mo --help .")
                openvino_model = KerasOpenVINOModel(model,
                                                    input_spec=input_spec,
                                                    precision=_precision,
                                                    thread_num=thread_num,
                                                    device=device,
                                                    config=openvino_config,
                                                    logging=logging,
                                                    **kwargs)
            if metric:
                if not isinstance(accuracy_criterion, dict):
                    accuracy_criterion = {'relative': 0.99, 'higher_is_better': True}
                drop_type = 'relative' if 'relative' in accuracy_criterion else 'absolute'
                higher_is_better = accuracy_criterion.get('higher_is_better', None)
                maximal_drop = accuracy_criterion.get(drop_type, None)
            else:
                drop_type, higher_is_better, maximal_drop = None, None, None
            result = openvino_model.pot(x=x,  # type: ignore
                                        y=y,
                                        metric=metric,
                                        higher_better=higher_is_better,
                                        drop_type=drop_type,
                                        maximal_drop=maximal_drop,
                                        max_iter_num=max_trials,
                                        sample_size=sample_size,
                                        config=openvino_config,
                                        thread_num=thread_num)
        elif accelerator == 'onnxruntime':
            # convert tensorflow model to onnx model
            from bigdl.nano.deps.onnxruntime.tensorflow.tensorflow_onnxruntime_model \
                import KerasONNXRuntimeModel
            if isinstance(model, KerasONNXRuntimeModel):     # type: ignore
                onnx_model = model
            else:
                onnx_model = InferenceOptimizer.trace(model=model, accelerator='onnxruntime',
                                                      input_spec=input_spec, thread_num=thread_num)

            # trace onnx model
            method_map = {
                'qlinear': 'onnxrt_qlinearops',
                'integer': 'onnxrt_integerops',
                None: 'onnxrt_qlinearops'  # default
            }
            framework = method_map.get(method, None)
            result = inc_quantzie(onnx_model, dataloader=(x, y),
                                  eval_func=eval_func,
                                  metric=metric,
                                  framework=framework,
                                  thread_num=thread_num,
                                  conf=conf,
                                  approach=approach,
                                  tuning_strategy=tuning_strategy,
                                  accuracy_criterion=accuracy_criterion,
                                  timeout=timeout,
                                  max_trials=max_trials,
                                  inputs=inputs,
                                  outputs=outputs,
                                  onnx_option='tensorflow',
                                  onnxruntime_session_options=onnxruntime_session_options)
            result._nesting_level = onnx_model._nesting_level
            result._inputs_dtypes = onnx_model._inputs_dtypes
            result._default_kwargs = onnx_model._default_kwargs
            result._call_fn_args_backup = onnx_model._call_fn_args_backup
            result._output_shape = onnx_model._output_shape
        else:
            invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))
        return patch_compiled_and_attrs(result, original_model)

    @staticmethod
    def save(model: Model, path):
        """
        Save the model to local file.

        :param model: Any model of keras.Model, including all models accelareted by
               InferenceOptimizer.trace/InferenceOptimizer.quantize.
        :param path: Path to saved model. Path should be a directory.
        """
        import yaml
        path = Path(path)
        path.mkdir(parents=path.parent, exist_ok=True)
        if hasattr(model, '_save'):
            model._save(path)
        else:
            # typically for keras Model
            meta_path = Path(path) / "nano_model_meta.yml"
            with open(meta_path, 'w+') as f:
                metadata = {
                    'ModelType': 'KerasModel',
                    'checkpoint': 'saved_weight.ckpt'
                }
                yaml.safe_dump(metadata, f)
            checkpoint_path = path / metadata['checkpoint']
            model.save(checkpoint_path)

    @staticmethod
    def load(path, model: Optional[Model] = None, device=None):
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
        :param device: A string represents the device of the inference. Default to None.
               Only valid for openvino model, otherwise will be ignored.
        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime) or
                 precision(FP32/FP16/BF16/INT8).
        """
        import yaml
        path = Path(path)
        invalidInputError(path.exists(), "{} doesn't exist.".format(path))
        meta_path = path / "nano_model_meta.yml"
        invalidInputError(meta_path.exists(),
                          "File {} is required to load model.".format(str(meta_path)))
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        model_type = metadata.get('ModelType', None)
        if model_type == 'KerasOpenVINOModel':
            result = load_openvino_model(path, framework='tensorflow', device=device)
            return patch_attrs(result, model)
        if model_type == 'KerasONNXRuntimeModel':
            result = load_onnxruntime_model(path, framework='tensorflow')
            return patch_attrs(result, model)
        if model_type == 'KerasQuantizedModel':
            result = load_inc_model(path, model, framework='tensorflow')
            return patch_attrs(result, model)
        if model_type == 'BF16Model':
            result = load_bf16_model(path)
            return patch_attrs(result, model)
        checkpoint_path = metadata.get('checkpoint', None)
        invalidInputError(checkpoint_path is not None, "Key 'checkpoint' must be specified.")
        checkpoint_path = path / metadata['checkpoint']
        model = keras.models.load_model(checkpoint_path)
        return model


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    for data_input, target in data:
        metric.update_state(y_true=target, y_pred=model(data_input))
    return metric.result().numpy()
