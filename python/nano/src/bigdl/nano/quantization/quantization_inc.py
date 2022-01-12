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
from abc import ABC, abstractmethod
from typing import Any
from neural_compressor.experimental.common import Metric

try:
    from neural_compressor.conf.config import Quantization_Conf
    from neural_compressor.experimental import Quantization
except ImportError:
    raise ImportError(
        "Module neural_compressor is not installed. Please install it by command: \"pip install "
        "neural-compressor\"")


class QuantizationINC(Quantization):
    def __init__(self,
                 framework: str,
                 conf='',
                 approach='post_training_static_quant',
                 tuning_strategy='bayesian',
                 accuracy_criterion: dict = None,
                 timeout=0,
                 max_trials=1,
                 inputs=None,
                 outputs=None
                 ):
        """
        Create a Intel Neural Compressor Quantization object. To understand INC quantization,
        please refer to https://github.com/intel/neural-compressor/blob/master/docs/Quantization.md.

        :param framework:   'tensorflow', 'pytorch', 'pytorch_fx', 'pytorch_ipex', 'onnxrt_integer',
                            'onnxrt_qlinear' or 'mxnet'; allow new framework backend extension.
                            Default: 'pytorch_fx'. Consistent with Intel Neural Compressor
                            Quantization.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, using default config.
        :param approach:    'post_training_static_quant', 'post_training_dynamic_quant',
                            'quant_aware_training'.
                            Default: 'post_training_static_quant'.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, higher_is_better=True}
                                     allows relative accuracy loss: 1%. accuracy_criterion = {
                                     'absolute': 0.99, higher_is_better=False} means accuracy
                                     < 0.99 must be satisfied.
        :param timeout:     Tuning timeout (seconds). Default: 0,  which means early stop.
                            combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: 1.
                            Combine with timeout field to decide when to exit.
        :param inputs:      For tensorflow to specify names of inputs. e.g. inputs=['img',]
        :param outputs:     For tensorflow to specify names of outputs. e.g. outputs=['logits',]
        """
        qconf = Quantization_Conf(conf)
        cfg = qconf.usr_cfg
        # Override default config
        cfg.model.framework = framework
        cfg.quantization.approach = approach
        cfg.tuning.strategy.name = tuning_strategy
        if accuracy_criterion:
            cfg.tuning.accuracy_criterion = accuracy_criterion
        cfg.tuning.exit_policy.timeout = timeout
        cfg.tuning.exit_policy.max_trials = max_trials
        cfg.model.inputs = inputs
        cfg.model.outputs = outputs
        super().__init__(qconf)

    def post_training_quantize(self, model, calib_dataloader=None, val_dataloader=None,
                               metric=None):
        self.check(calib_dataloader, val_dataloader, metric)
        self.model = model
        if calib_dataloader:
            self.calib_dataloader = calib_dataloader
        if val_dataloader:
            self.eval_dataloader = val_dataloader
            if metric.metric:
                self.metric = Metric(metric, name='INC_' + type(metric).__name__)

        quantized = self()
        if quantized:
            return quantized
        else:
            raise RuntimeError("Found no quantized model satisfying accuracy criterion.")

    def check(self, calib_dataloader, val_dataloader, metric):
        """
        Call before self.__call__() to check if the object is well-initialized
        for quantization.
        """
        if self.cfg.quantization.approach == 'post_training_static_quant':
            assert calib_dataloader, \
                "calib_calib_dataloader must not be None when approach is " \
                "post-training static quantization."

        if self.cfg.quantization.approach == 'post_training_dynamic_quant':
            assert calib_dataloader is None, \
                "calib_calib_dataloader must be None when approach is " \
                "post-training dynamic quantization."

        if metric.metric and not val_dataloader:
            raise RuntimeError("val_dataloader must be specified when metric is not None.")


class INCMetric(ABC):
    metric: Any = None

    def __init__(self):
        assert self.metric, "Class variable 'metric' must not be None.'"
        self.pred_list = []
        self.label_list = []

    def update(self, preds, labels):
        # add preds and labels to storage
        self.pred_list.extend(preds)
        self.label_list.extend(labels)

    def reset(self):
        # clear preds and labels storage
        self.pred_list = []
        self.label_list = []

    def result(self):
        # calculate accuracy
        preds, labels = self.stack(self.pred_list, self.label_list)
        accuracy = self.metric(preds, labels)
        return self.to_scalar(accuracy)

    @abstractmethod
    def stack(self, preds, labels):
        pass

    @abstractmethod
    def to_scalar(self, tensor):
        pass


class TorchINCMetric(INCMetric):
    def stack(self, preds, labels):
        import torch
        # calculate accuracy
        preds = torch.stack(preds)
        labels = torch.stack(labels)
        return preds, labels

    def to_scalar(self, tensor):
        return tensor.item()


class KerasINCMetric(INCMetric):
    def stack(self, preds, labels):
        import tensorflow as tf
        # calculate accuracy
        preds = tf.stack(preds)
        labels = tf.stack(labels)
        return preds, labels

    def to_scalar(self, tensor):
        return tensor.numpy()
