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


def quantize(self,
             precision: str = 'int8',
             accelerator: str = None,
             calib_dataset: tf.data.Dataset = None,
             metric: Metric = None,
             accuracy_criterion: dict = None,
             approach: str = 'static',
             method: str = None,
             conf: str = None,
             tuning_strategy: str = None,
             timeout: int = None,
             max_trials: int = None,
             batch=None,
             inputs: List[str] = None,
             outputs: List[str] = None):
    """
    Post-training quantization on a keras model.

    :param calib_dataset:   A tf.data.Dataset object for calibration. Required for
                            static quantization. It's also used as validation dataloader.
    :param precision:       Global precision of quantized model,
                            supported type: 'int8', 'bf16', 'fp16', defaults to 'int8'.
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
                            Default: 'static'. OpenVINO supports static mode only.
    :param method:      Method to do quantization. When accelerator=None, supported methods: None.
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
    :param batch:       Batch size of dataloader for calib_dataset. Defaults to None, if the
                        dataset is not a BatchDataset, batchsize equals to 1. Otherwise,
                        batchsize complies with the dataset._batch_size.
    :param inputs:      A list of input names.
                        Default: None, automatically get names from graph.
    :param outputs:     A list of output names.
                        Default: None, automatically get names from graph.
    :return:            A TensorflowBaseModel for INC. If there is no model found, return None.
    """
    if accelerator is None:
        if batch and calib_dataset:
            calib_dataset = calib_dataset.batch(batch)
        return inc_quantzie(self, dataloader=calib_dataset, metric=metric,
                            framework='tensorflow',
                            conf=conf,
                            approach=approach,
                            tuning_strategy=tuning_strategy,
                            accuracy_criterion=accuracy_criterion,
                            timeout=timeout,
                            max_trials=max_trials,
                            inputs=inputs,
                            outputs=outputs)
    else:
        invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))
