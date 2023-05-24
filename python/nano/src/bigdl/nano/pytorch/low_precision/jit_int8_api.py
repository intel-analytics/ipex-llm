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


def PytorchJITINT8Model(model, calib_data, q_config=None,
                        input_sample=None, channels_last=False,
                        thread_num=None, jit_strict=True,
                        jit_method=None, enable_onednn=False,
                        example_kwarg_inputs=None):
    '''
    :param model: the model(nn.module) to be transform if from_load is False
           the accelerated model if from_load is True.
    :param calib_data: calibration data is required for static quantization.
    :param q_config: We support 2 types of input here:

           | 1. Qconfig (https://pytorch.org/docs/stable/generated/torch.quantization.
           | qconfig.QConfig.html#qconfig) is the configuration for how we insert
           | observers for a particular operator. Quantization preparation function
           | will instantiate observers multiple times for each of the layers.
           |
           | 2. QConfigMapping (https://pytorch.org/docs/stable/generated/torch.ao.
           | quantization.qconfig_mapping.QConfigMapping.html#qconfigmapping)
           | (recommended) is a collection of quantization configurations, user
           | can set the qconfig for each operator (torch op calls, functional
           | calls, module calls) in the model through qconfig_mapping.

    :param input_sample: torch tensor indicate the data sample to be used
           for tracing.
    :param channels_last: if set model and data to be channels-last mode.
    :param thread_num: the thread num allocated for this model.
    :param from_load: this will only be set by _load method.
    :param jit_strict: Whether recording your mutable container types.
    :param jit_method: use ``jit.trace`` or ``jit.script`` to convert a model
           to TorchScript.
    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on
           oneDNN Graph API, which provides a flexible API for aggressive
           fusion. Default to ``False``.
    :param example_kwarg_inputs: keyword arguments of example inputs that will be passed
           to ``torch.jit.trace``. Default to None. Either this argument or input_sample
           should be specified when use_jit is ``True`` and torch > 2.0,
           sotherwise will be ignored.
    '''
    from .jit_int8_model import PytorchJITINT8Model
    return PytorchJITINT8Model(model, calib_data, q_config=q_config,
                               input_sample=input_sample, channels_last=channels_last,
                               thread_num=thread_num, jit_strict=jit_strict,
                               jit_method=jit_method, enable_onednn=enable_onednn,
                               example_kwarg_inputs=example_kwarg_inputs)


def load_pytorchjitint8_model(path):
    from .jit_int8_model import PytorchJITINT8Model
    return PytorchJITINT8Model._load(path)
