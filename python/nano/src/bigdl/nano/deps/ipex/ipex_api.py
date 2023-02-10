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


from typing import Any
from bigdl.nano.utils.common import invalidInputError


def ipex_optimize(model: Any, optimizers: Any = None, dtype: Any = None,
                  inplace: bool = False, weights_prepack: Any = None):
    import intel_extension_for_pytorch as ipex
    training = model.training
    if optimizers is not None and not isinstance(optimizers, (list, tuple)):
        model.train()
        optimizer = optimizers
    elif optimizers is None or len(optimizers) == 0:
        model.eval()
        optimizer = None
    elif len(optimizers) == 1:
        model.train()
        optimizer = optimizers[0]
    else:
        invalidInputError(False, "Ipex does not support more than one optimizers.")
    ret = ipex.optimize(model=model,
                        dtype=dtype,
                        optimizer=optimizer,
                        inplace=inplace,
                        weights_prepack=weights_prepack)
    model.train(training)

    return ret


def PytorchIPEXJITModel(model, input_sample=None, use_ipex=False,
                        use_jit=False, channels_last=None, thread_num=None,
                        inplace=False, jit_strict=True, jit_method=None,
                        weights_prepack=None, enable_onednn=True):
    '''
    :param model: the model(nn.module) to be transform.
    :param input_sample: torch tensor indicate the data sample to be used
            for tracing.
    :param use_ipex: if use ipex to optimize the model
    :param use_jit: if use jit to accelerate the model
    :param channels_last: if set model and data to be channels-last mode.
    :param thread_num: the thread num allocated for this model.
    :param inplace: whether to perform inplace optimization. Default: ``False``.
    :param jit_strict: Whether recording your mutable container types.
    :param jit_method: use ``jit.trace`` or ``jit.script`` to
           convert a model to TorchScript.
    :param weights_prepack: Whether to perform weight prepack for convolution and linear
           to avoid oneDNN weights reorder. The default value is None. Explicitly setting
           this knob overwrites the configuration set by level knob. Only valid when
           ``use_ipex=True``, otherwise will be ignored.
    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph
           API, which provides a flexible API for aggressive fusion. Default to
           ``True``, only valid when use_jit is ``True``, otherwise will be ignored.
    '''
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex,
                               use_jit=use_jit, channels_last=channels_last,
                               thread_num=thread_num, inplace=inplace, jit_strict=jit_strict,
                               jit_method=jit_method, weights_prepack=weights_prepack,
                               enable_onednn=enable_onednn)


def PytorchIPEXJITBF16Model(model, input_sample=None, use_ipex=False,
                            use_jit=False, channels_last=None, thread_num=None,
                            inplace=False, jit_strict=True, jit_method=None,
                            weights_prepack=None, enable_onednn=True):
    '''
    :param model: the model(nn.module) to be transform.
    :param input_sample: torch tensor indicate the data sample to be used
            for tracing.
    :param use_ipex: if use ipex to optimize the model
    :param use_jit: if use jit to accelerate the model
    :param channels_last: if set model and data to be channels-last mode.
    :param thread_num: the thread num allocated for this model.
    :param inplace: whether to perform inplace optimization. Default: ``False``.
    :param jit_strict: Whether recording your mutable container types.
    :param jit_method: use ``jit.trace`` or ``jit.script`` to
           convert a model to TorchScript.
    :param weights_prepack: Whether to perform weight prepack for convolution and linear
           to avoid oneDNN weights reorder. The default value is None. Explicitly setting
           this knob overwrites the configuration set by level knob. Only valid when
           ``use_ipex=True``, otherwise will be ignored.
    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph
           API, which provides a flexible API for aggressive fusion. Default to
           ``True``, only valid when use_jit is ``True``, otherwise will be ignored.
    '''
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model(model, input_sample=input_sample, use_ipex=use_ipex,
                                   use_jit=use_jit, channels_last=channels_last,
                                   thread_num=thread_num, inplace=inplace, jit_strict=jit_strict,
                                   jit_method=jit_method, weights_prepack=weights_prepack,
                                   enable_onednn=enable_onednn)


def PytorchIPEXQuantizationModel(model, calib_data, q_config=None,
                                 input_sample=None, channels_last=None,
                                 thread_num=None, inplace=False,
                                 jit_strict=True):
    '''
    :param model: the model(nn.module) to be transform.
    :param calib_data: calibration data is required for static quantization.
    :param q_config: describes how to quantize a layer or a part of the network
            by providing settings (observer classes) for activations and weights
            respectively.
    :param input_sample: torch tensor indicate the data sample to be used
            for tracing.
    :param channels_last: if set model and data to be channels-last mode.
    :param thread_num: the thread num allocated for this model.
    :param inplace: whether to perform inplace optimization. Default: ``False``.
    :param jit_strict: Whether recording your mutable container types.
    '''
    from .ipex_quantization_model import PytorchIPEXQuantizationModel
    return PytorchIPEXQuantizationModel(model, calib_data, q_config=q_config,
                                        input_sample=input_sample, channels_last=channels_last,
                                        thread_num=thread_num, inplace=inplace,
                                        jit_strict=jit_strict)


def load_ipexjit_model(path, model, inplace=False):
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel._load(path, model, inplace=inplace)


def load_ipexjitbf16_model(path, model, inplace=False):
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model._load(path, model, inplace=inplace)


def load_ipex_quantization_model(path, model, inplace=False):
    from .ipex_quantization_model import PytorchIPEXQuantizationModel
    return PytorchIPEXQuantizationModel._load(path, model, inplace=inplace)
