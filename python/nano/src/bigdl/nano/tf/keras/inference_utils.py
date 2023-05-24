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
from typing import Optional, List, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from bigdl.nano.utils.common import deprecated


class InferenceUtils:
    """A mixedin class for nano keras Sequential and Model, adding more functions for Inference."""

    @deprecated(func_name="model.quantize",
                message="Please use `bigdl.nano.tf.keras.InferenceOptimizer.quantize` instead.")
    def quantize(self,
                 x: Union[tf.Tensor, np.ndarray, tf.data.Dataset],
                 y: Union[tf.Tensor, np.ndarray] = None,
                 precision: str = 'int8',
                 accelerator: Optional[str] = None,
                 input_spec=None,
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
                 inputs: List[str] = None,
                 outputs: List[str] = None,
                 sample_size: int = 100,
                 onnxruntime_session_options=None,
                 openvino_config=None,
                 logging: bool = True):
        """
        Post-training quantization on a keras model.

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
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', defaults to 'int8'.
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', defaults to None.
                                None means staying in tensorflow.
        :param input_spec: (optional) A (tuple or list of) ``tf.TensorSpec``
                           defining the shape/dtype of the input. If ``accelerator='onnxruntime'``,
                           ``input_spec`` is required. If ``accelerator='openvino'``, or
                           ``accelerator=None`` and ``precision='int8'``, ``input_spec``
                           is required when you have a custom Keras model.
        :param metric:          A tensorflow.keras.metrics.Metric object for evaluation.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better': True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, 'higher_is_better':False} means accuracy
                                    must be smaller than 0.99.
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
        :return:            A TensorflowBaseModel for INC. If there is no model found, return None.

        .. warning::
           This function will be deprecated in future release.

           Please use ``bigdl.nano.tf.keras.InferenceOptimizer.quantize`` instead.
        """
        from bigdl.nano.tf.keras import InferenceOptimizer

        return InferenceOptimizer.quantize(
            model=self,
            x=x,
            y=y,
            precision=precision,
            accelerator=accelerator,
            input_spec=input_spec,
            metric=metric,
            accuracy_criterion=accuracy_criterion,
            approach=approach,
            method=method,
            conf=conf,
            tuning_strategy=tuning_strategy,
            timeout=timeout,
            max_trials=max_trials,
            batch=batch,
            thread_num=thread_num,
            inputs=inputs,
            outputs=outputs,
            sample_size=sample_size,
            onnxruntime_session_options=onnxruntime_session_options,
            openvino_config=openvino_config,
            logging=logging
        )

    @deprecated(func_name="model.trace",
                message="Please use `bigdl.nano.tf.keras.InferenceOptimizer.trace` instead.")
    def trace(self,
              accelerator: Optional[str] = None,
              input_spec=None,
              thread_num: Optional[int] = None,
              onnxruntime_session_options=None,
              openvino_config=None,
              logging=True):
        """
        Trace a Keras model and convert it into an accelerated module for inference.

        :param accelerator: The accelerator to use, defaults to None meaning staying in Keras
                            backend. 'openvino' and 'onnxruntime' are supported for now.
        :param input_spec: (optional) A (tuple or list of) ``tf.TensorSpec``
                           defining the shape/dtype of the input. If ``accelerator='onnxruntime'``,
                           ``input_spec`` is required. If ``accelerator='openvino'``,
                           ``input_spec`` is only required when you have a custom Keras model.
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param openvino_config: The config to be inputted in core.compile_model. Only valid when
                                accelerator='openvino', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. Default: ``True``.
        :return: Model with different acceleration(OpenVINO/ONNX Runtime).

        .. warning::
           This function will be deprecated in future release.

           Please use ``bigdl.nano.tf.keras.InferenceOptimizer.trace`` instead.
        """
        from bigdl.nano.tf.keras import InferenceOptimizer

        return InferenceOptimizer.trace(
            model=self,
            accelerator=accelerator,
            input_spec=input_spec,
            thread_num=thread_num,
            onnxruntime_session_options=onnxruntime_session_options,
            openvino_config=openvino_config,
            logging=logging
        )
