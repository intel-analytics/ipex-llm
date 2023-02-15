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


from abc import abstractmethod
from typing import Dict, Optional
from bigdl.nano.utils.common import invalidInputError, invalidOperationError

from .acceleration_option import AccelerationOption
from .format import format_acceleration_option
from .metric import CompareMetric


class BaseInferenceOptimizer:
    ALL_INFERENCE_ACCELERATION_METHOD = None

    def __init__(self):
        '''
        InferenceOptimizer for Pytorch/TF Model.

        It can be used to accelerate your model's inference speed
        with very few code changes.
        '''
        # optimized_model_dict handles the optimized model and some metadata
        # in {"method_name": {"latency": ..., "accuracy": ..., "model": ...}}
        self.optimized_model_dict = {}
        self._optimize_result = None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    def summary(self):
        '''
        Print format string representation for optimization result.
        '''
        invalidOperationError(len(self.optimized_model_dict) > 0,
                              "There is no optimization result. You should call .optimize() "
                              "before summary()")
        print(self._optimize_result)

    def get_model(self, method_name: str):
        """
        According to results of `optimize`, obtain the model with method_name.

        The available methods are "original", "fp32_channels_last", "fp32_ipex",
        "fp32_ipex_channels_last", "bf16", "bf16_channels_last", "bf16_ipex",
        "bf16_ipex_channels_last", "static_int8", "static_int8_ipex", "jit_fp32",
        "jit_fp32_channels_last", "jit_bf16", "jit_bf16_channels_last",
        "jit_fp32_ipex", "jit_fp32_ipex_channels_last", "jit_bf16_ipex",
        "jit_bf16_ipex_channels_last", "jit_int8", "jit_int8_channels_last",
        "openvino_fp32", "openvino_int8", "onnxruntime_fp32",
        "onnxruntime_int8_qlinear" and "onnxruntime_int8_integer".

        :param method_name: (optional) Obtain specific model according to method_name.
        :return: Model with different acceleration.
        """
        invalidOperationError(len(self.optimized_model_dict) > 0,
                              "There is no optimized model. You should call .optimize() "
                              "before get_model()")
        invalidInputError(method_name in self.ALL_INFERENCE_ACCELERATION_METHOD.keys(),
                          f"The model name you passed does not exist in the existing method "
                          f"list{list(self.ALL_INFERENCE_ACCELERATION_METHOD.keys())}, "
                          f"please re-enter the model name again.")
        invalidInputError("model" in self.optimized_model_dict[method_name],
                          "Unable to get the specified model as it doesn't exist in "
                          "optimized_model_dict.")
        return self.optimized_model_dict[method_name]["model"]

    def get_best_model(self,
                       accelerator: Optional[str] = None,
                       precision: Optional[str] = None,
                       use_ipex: Optional[bool] = None,
                       accuracy_criterion: Optional[float] = None):
        '''
        According to results of `optimize`, obtain the model with minimum latency under
        specific restrictions or without restrictions.

        :param accelerator: (optional) Use accelerator 'None', 'onnxruntime',
               'openvino', 'jit', defaults to None. If not None, then will only find the
               model with this specific accelerator.
        :param precision: (optional) Supported type: 'int8', 'bf16', and 'fp32'.
               Defaults to None which represents no precision limit. If not None, then will
               only find the model with this specific precision.
        :param use_ipex: (optional) if not None, then will only find the
               model with this specific ipex setting. This is only effective for pytorch model.
        :param accuracy_criterion: (optional) a float represents tolerable
               accuracy drop percentage, defaults to None meaning no accuracy control.
        :return: best model, corresponding acceleration option
        '''
        invalidOperationError(len(self.optimized_model_dict) > 0,
                              "There is no optimized model. You should call .optimize() "
                              "before get_best_model()")
        invalidInputError(accelerator in [None, 'onnxruntime', 'openvino', 'jit'],
                          "Only support accelerator 'onnxruntime', 'openvino' and 'jit'.")
        invalidInputError(precision in [None, 'int8', 'bf16', 'fp32'],
                          "Only support precision 'int8', 'bf16', 'fp32'.")
        if accuracy_criterion is not None and not self._calculate_accuracy:
            invalidInputError(False, "If you want to specify accuracy_criterion, you need "
                              "to set metric and validation_data when call 'optimize'.")

        best_model = self.optimized_model_dict["original"]["model"]
        best_metric = CompareMetric("original",
                                    self.optimized_model_dict["original"]["latency"],
                                    self.optimized_model_dict["original"]["accuracy"])

        has_limit = (accelerator is not None) or (precision is not None) or (use_ipex is not None)
        find_model = False

        for method in self.optimized_model_dict.keys():
            if method == "original" or self.optimized_model_dict[method]["status"] != "successful":
                continue
            option: AccelerationOption = self.ALL_INFERENCE_ACCELERATION_METHOD[method]
            result: Dict = self.optimized_model_dict[method]
            if accelerator is not None:
                if not getattr(option, accelerator):
                    continue
            if precision is not None:
                if precision == 'bf16' and not option.bf16:
                    continue
                if precision == 'int8' and not (option.inc or option.pot or option.fx):
                    continue
                if precision == 'fp32' and option.get_precision() != 'fp32':
                    continue
            if use_ipex:
                if not option.ipex:
                    continue
            find_model = True
            if accuracy_criterion is not None:
                accuracy = result["accuracy"]
                if isinstance(accuracy, str):
                    accuracy: float = self.optimized_model_dict["original"]["accuracy"]
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
                if not isinstance(result["accuracy"], str):
                    accuracy = result["accuracy"]
                else:
                    accuracy = self.optimized_model_dict["original"]["accuracy"]
                best_metric = CompareMetric(method, result["latency"], accuracy)

        if has_limit and not find_model:
            invalidInputError(False,
                              "Don't find related model in optimize's results.")

        return best_model, format_acceleration_option(best_metric.method_name,
                                                      self.ALL_INFERENCE_ACCELERATION_METHOD)
