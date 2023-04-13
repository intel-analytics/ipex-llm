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


from bigdl.nano.utils.common import invalidInputError
from .core.base_metric import BaseINCMetric
from bigdl.nano.utils.common import compare_version
import operator


def quantize(model, dataloader=None, eval_func=None, metric=None,
             thread_num=None, **kwargs):
    if kwargs['approach'] not in ['static', 'dynamic']:
        invalidInputError(False,
                          "Approach should be 'static' or 'dynamic', "
                          "{} is invalid.".format(kwargs['approach']))
    not_none_kwargs = {}
    for k, v in kwargs.items():
        # pop None values to use default
        if v is not None:
            not_none_kwargs[k] = v

    q_model = _quantize(model=model, dataloader=dataloader, eval_func=eval_func,
                        metric=metric, **not_none_kwargs)

    if 'pytorch' in not_none_kwargs['framework']:
        from .pytorch.quantized_model import PytorchQuantizedModel
        quantized_model = PytorchQuantizedModel(q_model, thread_num)

        from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object
        patch_attrs_from_model_to_object(model, quantized_model)
        return quantized_model

    elif 'tensorflow' in not_none_kwargs['framework']:
        from .tensorflow.model import KerasQuantizedModel
        return KerasQuantizedModel(q_model)

    elif 'onnx' in not_none_kwargs['framework']:
        onnxruntime_session_options = not_none_kwargs.get('onnxruntime_session_options', None)
        onnx_option = not_none_kwargs.get('onnx_option', None)
        if onnxruntime_session_options is None:
            import onnxruntime
            onnxruntime_session_options = onnxruntime.SessionOptions()
        if thread_num is not None:
            onnxruntime_session_options.intra_op_num_threads = thread_num
            onnxruntime_session_options.inter_op_num_threads = thread_num
        if onnx_option == 'tensorflow':
            from bigdl.nano.deps.onnxruntime.tensorflow.model import \
                KerasONNXRuntimeModel
            return KerasONNXRuntimeModel(q_model.model,
                                         onnxruntime_session_options=onnxruntime_session_options)
        else:
            from bigdl.nano.deps.onnxruntime.pytorch.pytorch_onnxruntime_model import \
                PytorchONNXRuntimeModel
            quantized_model = PytorchONNXRuntimeModel(q_model.model, None,
                                                      onnxruntime_session_options)

            from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object
            patch_attrs_from_model_to_object(model, quantized_model)
            return quantized_model

    else:
        invalidInputError(False,
                          f"Invalid framework argument: {not_none_kwargs['framework']}")


def _quantize(
    model,
    metric,
    dataloader,
    eval_func,
    framework,
    conf=None,
    approach='static',
    tuning_strategy='bayesian',
    accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
    timeout=0,
    max_trials=1,
    inputs=[],
    outputs=[],
    **kwargs
):
    from neural_compressor import quantization
    if compare_version("neural_compressor", operator.ge, "2.1"):
        from neural_compressor import PostTrainingQuantConfig
    else:
        from neural_compressor.quantization import PostTrainingQuantConfig
    from neural_compressor.config import AccuracyCriterion, TuningCriterion
    from neural_compressor.conf.pythonic_config import Config
    from neural_compressor.conf.config import QuantConf
    from neural_compressor.data import DataLoader

    if approach == "dynamic":
        dataloader = None
    elif 'onnx' in framework:
        if hasattr(dataloader, "batch_size"):
            # `dataloader` is pytorch DataLoader
            # we should convert pytorch tensor to numpy ndarray
            import torch

            def wrap_collate_fn(func):
                def new_collate_fn(batch):
                    return [x.detach().numpy() if isinstance(x, torch.Tensor) else x
                            for x in func(batch)]
                return new_collate_fn

            dataloader = DataLoader(framework, dataloader.dataset,
                                    collate_fn=wrap_collate_fn(dataloader.collate_fn))
        else:
            # `dataloader` is tensorflow (x,y)
            # we should construct a INC DataLoader from tf.Dataset or numpy ndarray
            import tensorflow as tf
            if isinstance(dataloader[0], tf.data.Dataset):
                dataloader = DataLoader("tensorflow", dataloader[0])
            else:
                from .onnx.tensorflow.quantization import KerasNumpyDataset
                dataloader = KerasNumpyDataset(dataloader[0], dataloader[1])

    elif not hasattr(dataloader, "batch_size") and not hasattr(dataloader, "_batch_size"):
        # INC requires a batched dataloader,
        # A torch.Dataset doesn't have `batch_size` attribute,
        # A tf.Dataset without `_batch_size` attribute is unbatched,
        # So we construct a INC DataLoader from them
        dataloader = DataLoader(framework, dataloader)

    if 'pytorch' in framework:
        # INC 1.14 and 2.0 doesn't support quantizing pytorch-lightning module for now
        from bigdl.nano.pytorch.lightning import LightningModule
        if isinstance(model, LightningModule):
            model = model.model
        if metric is not None:
            from .pytorch.metric import PytorchINCMetric
            inc_metric = PytorchINCMetric()
            inc_metric.metric = metric
    elif 'onnx' in framework:
        onnx_option = kwargs.pop('onnx_option', None)
        model = model.onnx_model
        if metric is not None:
            if onnx_option == 'tensorflow':
                from .onnx.tensorflow.metric import KerasONNXRuntimeINCMetic
                inc_metric = KerasONNXRuntimeINCMetic()
            else:
                from .onnx.pytorch.metric import PytorchONNXRuntimeINCMetic
                inc_metric = PytorchONNXRuntimeINCMetic()
            inc_metric.metric = metric
    elif 'tensorflow' in framework:
        if metric is not None:
            from .tensorflow.metric import TensorflowINCMetric
            inc_metric = TensorflowINCMetric()
            inc_metric.metric = metric

    custom_inc_metric = None
    # use metric_id and new custom_inc_metric to solve the same registered metric name
    if metric is not None:
        metric_id = str(inc_metric.get_next_metric_id())
        if 'tensorflow' in framework:
            # works for Tensorflow
            custom_inc_metric = type(type(inc_metric).__name__ + metric_id,
                                     (object, ),
                                     {"stack": inc_metric.stack,
                                      "to_scalar": inc_metric.to_scalar,
                                      "result": inc_metric.result,
                                      "update": inc_metric.update,
                                      "stack": inc_metric.stack,
                                      "reset": inc_metric.reset})
        else:
            # works for PyTorch
            custom_inc_metric = type(type(inc_metric).__name__ + metric_id,
                                     (BaseINCMetric, ),
                                     {"stack": inc_metric.stack,
                                      "to_scalar": inc_metric.to_scalar})
        custom_inc_metric = custom_inc_metric()
        custom_inc_metric.metric = metric

    if 'relative' in accuracy_criterion:
        criterion = 'relative'
    elif 'absolute' in accuracy_criterion:
        criterion = 'absolute'
    else:
        criterion = None

    if criterion is None:
        accuracy_criterion = AccuracyCriterion(
            higher_is_better=accuracy_criterion.get('higher_is_better', True),
        )
    else:
        accuracy_criterion = AccuracyCriterion(
            higher_is_better=accuracy_criterion.get('higher_is_better', True),
            criterion=criterion,
            tolerable_loss=float(accuracy_criterion[criterion])
        )

    tuning_criterion = TuningCriterion(
        strategy=tuning_strategy,
        timeout=timeout,
        max_trials=max_trials,
    )
    # For ipex quantization, we should set backend = "ipex"
    backend = "default" if framework != "pytorch_ipex" else "ipex"
    config = PostTrainingQuantConfig(
        accuracy_criterion=accuracy_criterion,
        tuning_criterion=tuning_criterion,
        approach=approach,
        inputs=inputs,
        outputs=outputs,
        backend=backend
    )
    if metric is None and eval_func is None:
        config.performance_only = True

    if compare_version("neural_compressor", operator.ge, "2.1"):
        q_conf = config
        # TODO: how to pass framework?
    else:
        config = Config(quantization=config, benchmark=None, pruning=None,
                        distillation=None, nas=None)

        q_conf = QuantConf()
        q_conf.map_pyconfig_to_cfg(config)
        q_conf.usr_cfg.model.framework = framework

    q_model = quantization.fit(
        model=model,
        conf=q_conf,
        calib_dataloader=dataloader,
        eval_func=eval_func,
        eval_metric=custom_inc_metric,
        # use same dataloader as 1.0 API
        eval_dataloader=dataloader if metric is not None else None,
    )
    return q_model
