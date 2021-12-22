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
                            Default: '', use default config.
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
        if conf:
            qconf = Quantization_Conf(conf)
        else:
            qconf = Quantization_Conf('')
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


class TorchMetricForINC:
    def __init__(self, metric):
        self.metric = metric
        self.pred_list = []
        self.label_list = []

    def update(self, preds, labels):
        # add preds and labels to storage
        self.pred_list.extend(preds)
        self.label_list.extend(labels)

    def reset(self):
        """clear preds and labels storage"""
        self.pred_list = []
        self.label_list = []

    def result(self):
        import torch
        # calculate accuracy
        preds = torch.stack(self.pred_list)
        labels = torch.stack(self.label_list)
        accuracy = self.metric(preds, labels)
        return accuracy.item()
