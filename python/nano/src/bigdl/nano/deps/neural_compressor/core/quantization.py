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
from bigdl.nano.utils.log4Error import invalidInputError
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.experimental import Quantization, common


class BaseQuantization(Quantization):
    def __init__(self,
                 framework: str,
                 conf=None,
                 approach='post_training_static_quant',
                 tuning_strategy='bayesian',
                 accuracy_criterion: dict = {'relative': 0.99, 'higher_is_better': True},
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
        self.validate_config(framework=framework, approach=approach,
                             tuning_strategy=tuning_strategy,
                             accuracy_criterion=accuracy_criterion,
                             timeout=timeout, max_trials=max_trials,
                             inputs=inputs, outputs=outputs)
        cfg = qconf.usr_cfg
        # Override default config
        cfg.model.framework = framework
        cfg.quantization.approach = approach
        cfg.tuning.strategy.name = tuning_strategy
        cfg.tuning.accuracy_criterion = accuracy_criterion
        cfg.tuning.exit_policy.timeout = timeout
        cfg.tuning.exit_policy.max_trials = max_trials
        cfg.model.inputs = inputs
        cfg.model.outputs = outputs
        super().__init__(qconf)

    def post_training_quantize(self, model, calib_dataloader=None, metric=None):
        self.sanity_check_before_execution(model, calib_dataloader, metric)
        model, calib_dataloader, metric = self._pre_execution(model, calib_dataloader,
                                                              metric)
        q_model = self._execution(model, calib_dataloader, metric)
        return self._post_execution(q_model)

    def _pre_execution(self, model, calib_dataloader, metric):
        return model, calib_dataloader, metric

    def _execution(self, model, calib_dataloader, metric):
        self.model = common.Model(model)

        class MyMetric(self._inc_metric_cls):
            def __init__(self):
                """
                This local class is to resolve dumping issue in tensorflow.
                In tensorflow, INC will try to dump the metric to yaml which
                somehow causes unexpected error. So we moved metric assignment
                to the new local class to avoid that.
                """
                self.metric = metric

        metric_kwargs = None
        if metric:
            metric_name = type(metric).__name__
            metric_id = self._inc_metric_cls.get_next_metric_id()
            metric_kwargs = {
                "metric_cls": MyMetric,
                "name": f"pytorch_{metric_name}_{metric_id}"
            }

        if self.cfg.quantization.approach == 'post_training_static_quant':
            self.calib_dataloader = calib_dataloader
        if metric_kwargs:
            self.eval_dataloader = calib_dataloader
            self.metric = common.Metric(**metric_kwargs)
        quantized = self()

        if quantized:
            return quantized
        else:
            invalidInputError(False,
                              "Found no quantized model satisfying accuracy criterion.")

    def _post_execution(self, q_model):
        return q_model

    def sanity_check_before_execution(self, model, calib_dataloader, metric):
        """
        Call before self.__call__() to check if the object is well-initialized
        for quantization.
        """
        if self.cfg.quantization.approach == 'post_training_static_quant':
            invalidInputError(calib_dataloader,
                              "calib_calib_dataloader must not be None"
                              " when approach is post-training static quantization.")

        if self.cfg.quantization.approach == 'post_training_dynamic_quant':
            if not metric:
                invalidInputError(calib_dataloader is None,
                                  "calib_calib_dataloader should be None when approach is"
                                  " post-training dynamic quantization and metric is None.")

        if metric and not calib_dataloader:
            invalidInputError(False,
                              "calib_dataloader must be specified for evaluation"
                              " when metric is not None.")

    @property
    def valid_frameworks(self):
        return ('pytorch_fx', 'pytorch', 'pytorch_ipex', 'tensorflow', 'onnxrt_integer',
                'onnxrt_qlinear')

    @property
    def valid_approaches(self):
        return 'post_training_static_quant', 'post_training_dynamic_quant'

    @property
    def valid_tuning_strategies(self):
        return 'basic', 'bayesian', 'mse', 'sigopt'

    @property
    def valid_drop_type(self):
        return 'relative', 'absolute'

    def validate_config(self, **kwargs):
        invalidInputError(
            kwargs['approach'] in self.valid_approaches,
            errMsg="{} is invalid.".format(kwargs['approach']),
            fixMsg="Choose from {}.".format(self.valid_approaches)
        )
        invalidInputError(
            kwargs['framework'] in self.valid_frameworks,
            errMsg="{} is invalid.".format(kwargs['framework']),
            fixMsg="Choose from {}.".format(self.valid_frameworks)
        )
        invalidInputError(
            kwargs['tuning_strategy'] in self.valid_tuning_strategies,
            errMsg="{} is invalid.".format(kwargs['tuning_strategy']),
            fixMsg="Choose from {}.".format(self.valid_tuning_strategies)
        )

        invalidInputError(
            isinstance(kwargs['accuracy_criterion'], dict),
            errMsg="accuracy_criterion should be a dictionary."
        )
        is_drop_type_defined = False
        for k, v in kwargs['accuracy_criterion'].items():
            if k in self.valid_drop_type:
                # either of 'relative'/'absolute' can be in the dict
                invalidInputError(
                    not is_drop_type_defined,
                    errMsg="Drop type should be defined only once.",
                    fixMsg="There multiple 'relative'/'absolute' in "
                           "accuracy_criterion, only one is required."
                )
                is_drop_type_defined = True
                invalidInputError(
                    0 <= v < 1,
                    errMsg="The value of {} for {} drop is invalid.".format(v, k),
                    fixMsg="Value should be within [0, 1)."
                )
            else:
                invalidInputError(
                    k == 'higher_is_better',
                    errMsg="Key {} is not valid".format(k),
                    fixMsg="Keys for accuracy_criterion should be"
                           " 'relative'/'absolute' and 'higher_is_better'."
                )

        invalidInputError(
            isinstance(kwargs['timeout'], int) and kwargs['timeout'] >= 0,
            errMsg="Argument timeout should be an integer >= 0"
        )
        invalidInputError(
            isinstance(kwargs['max_trials'], int) and kwargs['max_trials'] >= 0,
            errMsg="Argument max_trials should be an integer >= 0"
        )
