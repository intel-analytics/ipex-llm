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
                        use_jit=False, channels_last=None):
    '''
    :param model: the model(nn.module) to be transform.
    :param input_sample: torch tensor indicate the data sample to be used
            for tracing.
    :param use_ipex: if use ipex to optimize the model
    :param use_jit: if use jit to accelerate the model
    :param channels_last: if set model and data to be channels-last mode.
            the parameter will be ignored if use_ipex is False.
    '''
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex,
                               use_jit=use_jit, channels_last=channels_last)


def PytorchIPEXJITBF16Model(model, input_sample=None, use_ipex=False,
                            use_jit=False, channels_last=None):
    '''
    :param model: the model(nn.module) to be transform.
    :param input_sample: torch tensor indicate the data sample to be used
            for tracing.
    :param use_ipex: if use ipex to optimize the model
    :param use_jit: if use jit to accelerate the model
    :param channels_last: if set model and data to be channels-last mode.
            the parameter will be ignored if use_ipex is False.
    '''
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model(model, input_sample=input_sample, use_ipex=use_ipex,
                                   use_jit=use_jit, channels_last=channels_last)


def load_ipexjit_model(path, model):
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel._load(path, model)
