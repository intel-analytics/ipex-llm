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
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.experimental import Quantization
from neural_compressor.experimental.common import Metric

from .metric import METRICS


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
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better':True}
                                     allows relative accuracy loss: 1%. accuracy_criterion = {
                                     'absolute': 0.99, 'higher_is_better':False} means accuracy
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
            if metric:
                framework = self.cfg.model.framework
                if 'pytorch' in framework:
                    framework_metric = METRICS['pytorch']
                else:
                    framework_metric = METRICS[framework]

                class MyMetric(framework_metric):
                    def __init__(self):
                        """
                        This local class is to resolve dumping issue in tensorflow.
                        In tensorflow, INC will try to dump the metric to yaml which
                        somehow causes unexpected error. So we moved metric assignment
                        to the new local class to avoid that.
                        """
                        self.metric = metric

                self.metric = Metric(
                    MyMetric,
                    name="%s_%s" % (framework_metric.__name__, type(metric).__name__)
                )

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

        if metric and not val_dataloader:
            raise RuntimeError("val_dataloader must be specified when metric is not None.")
