import copy

from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.experimental import Quantization


class QuantizationINC:
    def __init__(self,
                 framework,
                 conf=None,
                 approach='post_training_static_quant',
                 strategy='bayesian',
                 accuracy_criterion=None,
                 timeout=0,
                 max_trials=1,
                 inputs=None,
                 outputs=None
                 ):
        """
        Create a Intel Neural Compressor Quantization object. To understand INC quantization,
        please refer to https://github.com/intel/neural-compressor/blob/master/docs/Quantization.md.

        :param framework:   Supported values are tensorflow, pytorch, pytorch_fx, pytorch_ipex,
                            onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend
                            extension. Default: pytorch_fx. Consistent with Intel Neural Compressor
                            Quantization.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, use default config.
        :param approach:    post_training_static_quant, post_training_dynamic_quant,
                            quant_aware_training. Default: post_training_static_quant.
        :param strategy:    bayesian, basic, mse, sigopt. Default: bayesian.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, higher_is_better=True}
                                     allows relative
                                    accuracy loss: 1%. accuracy_criterion = {'absolute': 0.99,
                                    higher_is_better=Flase} means accuracy < 0.99 must be satisfied.
        :param timeout:     Tuning timeout (seconds). Default: 0,  which means early stop.
                            combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: 1.
                            combine with timeout field to decide when to exit.
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
            cfg.tuning.strategy.name = strategy
            if accuracy_criterion:
                cfg.tuning.accuracy_criterion = accuracy_criterion
            cfg.tuning.exit_policy.timeout = timeout
            cfg.tuning.exit_policy.max_trials = max_trials
            if inputs:
                cfg.model.inputs = inputs
            if outputs:
                cfg.model.outputs = outputs
        self.quantizer = Quantization(qconf)

    def quantize_torch(self, model, calib_dataloader, val_dataloader, trainer,
                       metric: str):
        q_litmodel = copy.deepcopy(model)
        quantizer = self.quantizer
        quantizer.model = q_litmodel.model
        q_approach = quantizer.cfg['quantization']['approach']
        assert val_dataloader, "val_dataloader must be specified when tune=True."

        def eval_func(model_to_eval):
            q_litmodel.model = model_to_eval
            val_outputs = trainer.validate(q_litmodel, val_dataloader)
            return val_outputs[0][metric]
        quantizer.eval_func = eval_func

        if q_approach == 'quant_aware_training':
            def q_func(model_to_train):
                q_litmodel.model = model_to_train
                trainer.fit(q_litmodel, train_dataloaders=calib_dataloader)
            quantizer.q_func = q_func
        else:
            quantizer.calib_dataloader = calib_dataloader
        quantized = quantizer()
        if quantized:
            q_litmodel.model = quantized.model
            return q_litmodel
        return None

    def __call__(self, model, calib_dataloader, val_dataloader, *args, **kwargs):
        framework = self.quantizer.cfg.model.framework
        if 'pytorch' in framework:
            return self.quantize_torch(model, calib_dataloader, val_dataloader,  *args, **kwargs)
        else:
            raise NotImplementedError("Only support pytorch for now.")
