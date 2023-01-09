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


def quantize(model, dataloader=None, metric=None, thread_num=None, **kwargs):
    if kwargs['approach'] not in ['static', 'dynamic']:
        invalidInputError(False,
                          "Approach should be 'static' or 'dynamic', "
                          "{} is invalid.".format(kwargs['approach']))
    not_none_kwargs = {}
    for k, v in kwargs.items():
        # pop None values to use default
        if v is not None:
            not_none_kwargs[k] = v

    q_model = _quantize(model=model, dataloader=dataloader, metric=metric, **not_none_kwargs)

    if 'pytorch' in not_none_kwargs['framework']:
        from .pytorch.quantized_model import PytorchQuantizedModel
        quantized_model = PytorchQuantizedModel(q_model, thread_num)

        from bigdl.nano.pytorch.utils import patch_attrs_from_model_to_object
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
            from bigdl.nano.deps.onnxruntime.tensorflow.tensorflow_onnxruntime_model import \
                KerasONNXRuntimeModel
            return KerasONNXRuntimeModel(q_model.model,
                                         onnxruntime_session_options=onnxruntime_session_options)
        else:
            from bigdl.nano.deps.onnxruntime.pytorch.pytorch_onnxruntime_model import \
                PytorchONNXRuntimeModel
            quantized_model = PytorchONNXRuntimeModel(q_model.model, None,
                                                      onnxruntime_session_options)

            from bigdl.nano.pytorch.utils import patch_attrs_from_model_to_object
            patch_attrs_from_model_to_object(model, quantized_model)
            return quantized_model

    else:
        invalidInputError(False,
                          f"Invalid framework argument: {not_none_kwargs['framework']}")


def _quantize(
    model,
    metric,
    dataloader,
    framework,
    conf=None,
    approach='static',
    tuning_strategy='bayesian',
    accuracy_criterion={'relative': 0.99, 'higher_is_better': True},
    timeout=0,
    max_trials=1,
    inputs=[],
    outpus=[],
    **kwargs
):
    from neural_compressor import quantization
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
            import torch

            def wrap_collate_fn(func):
                def new_collate_fn(batch):
                    return [x.numpy() if isinstance(x, torch.Tensor) else x for x in func(batch)]
                return new_collate_fn

            dataloader = DataLoader(framework, dataloader.dataset,
                                    collate_fn=wrap_collate_fn(dataloader.collate_fn))
        else:
            # `dataloader` is tensorflow (x,y)
            from .onnx.tensorflow.quantization import KerasNumpyDataset
            dataloader = KerasNumpyDataset(dataloader[0], dataloader[1], model.dtype)
    elif not hasattr(dataloader, "batch_size") and not hasattr(dataloader, "_batch_size"):
        dataloader = DataLoader(framework, dataloader)

    model = model.onnx_model if 'onnx' in framework else model

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
            tolerable_loss=1.0 - accuracy_criterion[criterion]
        )

    tuning_criterion = TuningCriterion(
        strategy=tuning_strategy,
        timeout=timeout,
        max_trials=max_trials,
    )
    config = PostTrainingQuantConfig(
        accuracy_criterion=accuracy_criterion,
        tuning_criterion=tuning_criterion,
        approach=approach,
        inputs=inputs,
        outputs=outpus,
    )
    config.performance_only = True
    config = Config(quantization=config, benchmark=None, pruning=None,
                    distillation=None, nas=None)

    q_conf = QuantConf()
    q_conf.map_pyconfig_to_cfg(config)
    q_conf.usr_cfg.model.framework = framework

    q_model = quantization.fit(
        model=model,
        conf=q_conf,
        calib_dataloader=dataloader,
        # todo: add eval
    )
    return q_model
