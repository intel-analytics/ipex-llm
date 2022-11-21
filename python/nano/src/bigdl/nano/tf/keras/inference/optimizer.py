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
import time
import tensorflow as tf
from typing import Dict, Optional, List
from bigdl.nano.utils.inference.common.base_optimizer import BaseInferenceOptimizer
from bigdl.nano.utils.inference.common.checker import available_acceleration_combination
from bigdl.nano.utils.inference.common.utils import AccelerationOption,\
    throughput_calculate_helper, format_optimize_result
from bigdl.nano.tf.keras import Model as NanoModel
from bigdl.nano.utils.log4Error import invalidInputError
from tensorflow.keras import Model as Model
from tensorflow.data import Dataset
from tensorflow.keras.metrics import Metric


class TFAccelerationOption(AccelerationOption):
    def optimize(self, model, training_data=None, input_sample=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        accelerator = self.get_accelerator()
        if self.get_precision() == "fp32":
            # trace
            if accelerator is None:
                return model
            else:
                acce_model = model.trace(accelerator=accelerator,
                                         input_sample=input_sample,
                                         thread_num=thread_num,
                                         # remove output of openvino
                                         logging=logging)
        else:
            # quantize
            ort_method: str = self.method
            acce_model = model.quantize(precision=self.get_precision(),
                                        accelerator=accelerator,
                                        calib_dataset=training_data,
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
            "int8": TFAccelerationOption(inc=True),
            "openvino_fp32": TFAccelerationOption(openvino=True),
            "openvino_int8": TFAccelerationOption(openvino=True, pot=True),
            "onnxruntime_fp32": TFAccelerationOption(onnxruntime=True),
            "onnxruntime_int8_qlinear": TFAccelerationOption(onnxruntime=True, inc=True,
                                                             method="qlinear"),
            "onnxruntime_int8_integer": TFAccelerationOption(onnxruntime=True, inc=True,
                                                             method="integer"),
        }  # type: ignore

    def optimize(self, model: Model,
                 training_data: Dataset,
                 validation_data: Optional[Dataset] = None,
                 batch_size: int = 1,
                 metric: Optional[Metric] = None,
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

        The available methods are "original", "openvino_fp32", "onnxruntime_fp32", "int8".

        :param model: A keras.Model to be optimized
        :param training_data: An unbatched tf.data.Dataset object which is used for training.
                              This dataset will be used as calibration dataset for
                              Post-Training Static Quantization (PTQ), as well as be used for
                              generating input_sample to calculate latency.
                              To avoid data leak during calibration, please use training
                              dataset as much as possible.
        :param validation_data: (optional) An unbatched tf.data.Dataset object for accuracy
               evaluation. This is only needed when users care about the possible accuracy drop.
        :param metric: (optional) A tensorflow.keras.metrics.Metric object which is used for
               calculating accuracy.
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
        invalidInputError(isinstance(model, Model), "model should be a Keras Model.")
        invalidInputError(direction in ['min', 'max'],
                          "Only support direction 'min', 'max'.")

        if not isinstance(model, NanoModel):
            # turn model into NanoModel to obtain trace and quantize method
            model = NanoModel(inputs=model.inputs, outputs=model.outputs)

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
            # TODO: how to get and control thread num in tf?
            default_threads = None  # type: ignore
        thread_num = default_threads if thread_num is None else int(thread_num)  # type: ignore

        result_map: Dict[str, Dict] = {}

        batched_training_data = training_data.batch(batch_size)
        input_sample = next(iter(batched_training_data))
        # TODO: how to obtain input from output of training_data
        input_sample = input_sample[:-1]

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
                precision: str = option.get_precision()
                try:
                    acce_model = option.optimize(model=model,
                                                 training_data=training_data,
                                                 input_sample=tf.TensorSpec(
                                                     shape=input_sample.shape,
                                                     dtype=tf.float32),
                                                 thread_num=thread_num,
                                                 logging=logging,
                                                 sample_size_for_pot=sample_size_for_pot)
                except Exception as e:
                    print(e)
                    result_map[method]["status"] = "fail to convert"
                    print(f"----------Failed to convert to {method}----------")
                    continue

                result_map[method]["status"] = "successful"

                def func_test(model, sample):
                    model(sample)
                try:
                    result_map[method]["latency"], status =\
                        throughput_calculate_helper(latency_sample_num, baseline_time,
                                                    func_test, acce_model, input_sample)
                    if status is False and method != "original":
                        result_map[method]["status"] = "early stopped"
                        continue
                except Exception as e:
                    print(e)
                    result_map[method]["status"] = "fail to forward"
                    print(f"----------{method} failed to forward----------")
                    continue

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
                                                               batched_validation_data)
                            except Exception as e:
                                print(e)
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
        # save time cost to self._optimize_result
        time_cost = time.perf_counter() - start_time
        time_cost_str = f"Optimization cost {time_cost:.1f}s in total."
        self._optimize_result += time_cost_str
        print(self._optimize_result)
        print("===========================Stop Optimization===========================")


def _accuracy_calculate_helper(model, metric, data):
    '''
    A quick helper to calculate accuracy
    '''
    for data_input, target in data:
        metric.update_state(y_true=target, y_pred=model(data_input))
    return metric.result().numpy()
