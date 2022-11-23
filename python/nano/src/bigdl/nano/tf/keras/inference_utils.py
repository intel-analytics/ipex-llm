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
from typing import List
from bigdl.nano.deps.neural_compressor.inc_api import quantize as inc_quantzie
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.deps.openvino.openvino_api import KerasOpenVINOModel
from bigdl.nano.deps.onnxruntime.onnxruntime_api import KerasONNXRuntimeModel
from typing import Optional, List


class InferenceUtils:
    """A mixedin class for nano keras Sequential and Model, adding more functions for Inference."""

    def quantize(self,
                 calib_dataset: tf.data.Dataset,
                 precision: str = 'int8',
                 accelerator: Optional[str] = None,
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

        :param calib_dataset:   An unbatched tf.data.Dataset object for calibration.
                                Required for static quantization.
                                It's also used as validation dataloader.
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', defaults to 'int8'.
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', defaults to None.
                                None means staying in tensorflow.
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
        """
        invalidInputError(approach == 'static', "Only 'static' approach is supported now.")
        if accelerator is None:
            if batch:
                calib_dataset = calib_dataset.batch(batch)
            return inc_quantzie(self, dataloader=calib_dataset,
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
        elif accelerator == 'openvino':
            from bigdl.nano.deps.openvino.tf.model import KerasOpenVINOModel    # type: ignore
            if isinstance(self, KerasOpenVINOModel):    # type: ignore
                openvino_model = self
            else:
                openvino_model = self.trace(accelerator='openvino',
                                            thread_num=thread_num,
                                            logging=logging,
                                            openvino_config=openvino_config)
            if metric:
                if not isinstance(accuracy_criterion, dict):
                    accuracy_criterion = {'relative': 0.99, 'higher_is_better': True}
                drop_type = 'relative' if 'relative' in accuracy_criterion else 'absolute'
                higher_is_better = accuracy_criterion.get('higher_is_better', None)
                maximal_drop = accuracy_criterion.get(drop_type, None)
            else:
                drop_type, higher_is_better, maximal_drop = None, None, None
            return openvino_model.pot(dataset=calib_dataset.batch(1),    # type: ignore
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
            if isinstance(self, KerasONNXRuntimeModel):     # type: ignore
                onnx_model = self
            else:
                spec = tf.TensorSpec((self.input_shape), self.dtype)    # type: ignore
                onnx_model = self.trace(accelerator='onnxruntime',
                                        input_sample=spec,
                                        thread_num=thread_num)

            # trace onnx model
            method_map = {
                'qlinear': 'onnxrt_qlinearops',
                'integer': 'onnxrt_integerops',
                None: 'onnxrt_qlinearops'  # default
            }
            framework = method_map.get(method, None)
            return inc_quantzie(onnx_model, dataloader=calib_dataset.batch(1),
                                metric=metric,
                                framework=framework,
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
        else:
            invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))

    def trace(self,
              accelerator: Optional[str] = None,
              input_sample=None,
              thread_num: Optional[int] = None,
              onnxruntime_session_options=None,
              openvino_config=None,
              logging=True):
        """
        Trace a Keras model and convert it into an accelerated module for inference.

        :param input_sample: A set of inputs for trace, defaults to None. It should be a
            (tuple or list of) tf.TensorSpec or numpy array defining the shape/dtype of the input
            when using 'onnxruntime' accelerator. The parameter will be ignored if accelerator
            is 'openvino'.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Keras
                            backend. 'openvino' and 'onnxruntime' are supported for now.
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
        """
        if accelerator == 'openvino':
            final_openvino_option = {"INFERENCE_PRECISION_HINT": "f32"}
            if openvino_config is not None:
                final_openvino_option.update(openvino_config)
            return KerasOpenVINOModel(self,
                                      thread_num=thread_num,
                                      config=final_openvino_option,
                                      logging=logging)
        elif accelerator == 'onnxruntime':
            if onnxruntime_session_options is None:
                import onnxruntime
                onnxruntime_session_options = onnxruntime.SessionOptions()
                if thread_num is not None:
                    onnxruntime_session_options.intra_op_num_threads = thread_num
                    onnxruntime_session_options.inter_op_num_threads = thread_num
            return KerasONNXRuntimeModel(self, input_sample, onnxruntime_session_options)
        return self
