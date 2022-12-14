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

def create_IPEXAccelerator(*args, **kwargs):
    from bigdl.nano.deps.ipex.ipex_accelerator import IPEXAccelerator
    return IPEXAccelerator(*args, **kwargs)


def create_IPEXAccelerator_1_9(*args, **kwargs):
    from bigdl.nano.deps.ipex.version_1_9.ipex_accelerator_1_9 import IPEXAccelerator
    return IPEXAccelerator(*args, **kwargs)


def ipex_optimize(*args, **kwargs):
    import intel_extension_for_pytorch as ipex
    ipex.optimize(*args, **kwargs)


def ipex_device():
    from bigdl.nano.deps.ipex.version_1_9 import DEVICE
    return DEVICE


def PytorchIPEXJITModel(model, input_sample=None, use_ipex=False,
                        use_jit=False, channels_last=None, thread_num=None,
                        inplace=False, jit_strict=True, jit_method=None):
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
    '''
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex,
                               use_jit=use_jit, channels_last=channels_last,
                               thread_num=thread_num, inplace=inplace, jit_strict=jit_strict,
                               jit_method=jit_method)


def PytorchIPEXJITBF16Model(model, input_sample=None, use_ipex=False,
                            use_jit=False, channels_last=None, thread_num=None,
                            inplace=False, jit_strict=True, jit_method=None):
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
    '''
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model(model, input_sample=input_sample, use_ipex=use_ipex,
                                   use_jit=use_jit, channels_last=channels_last,
                                   thread_num=thread_num, inplace=inplace, jit_strict=jit_strict,
                                   jit_method=jit_method)


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
