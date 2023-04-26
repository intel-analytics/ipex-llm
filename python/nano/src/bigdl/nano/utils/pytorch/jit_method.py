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

import operator
import torch
from ..common import compare_version


def jit_convert(model, input_sample, jit_method=None,
                jit_strict=True, example_kwarg_inputs=None):
    '''
    Internal function to export pytorch model to TorchScript.

    :param model: the model(nn.module) to be transform
    :param input_sample: torch.Tensor or a list for the model tracing.
    :param jit_method: use ``jit.trace`` or ``jit.script`` to convert a model
        to TorchScript.
    :param jit_strict: Whether recording your mutable container types.
    :param example_kwarg_inputs: keyword arguments of example inputs that will be
        passed to ``torch.jit.trace``. Default to ``None``. Either this argument or
        ``input_sample`` should be specified when use_jit is ``True`` and torch > 2.0,
        otherwise will be ignored.
    '''
    if jit_method == 'trace':
        if compare_version("torch", operator.ge, "2.0"):
            model = torch.jit.trace(
                model,
                example_inputs=input_sample,
                check_trace=False,
                strict=jit_strict,
                example_kwarg_inputs=example_kwarg_inputs)
        else:
            model = torch.jit.trace(
                model, input_sample,
                check_trace=False,
                strict=jit_strict)
    elif jit_method == 'script':
        model = torch.jit.script(model)
    else:
        try:
            if compare_version("torch", operator.ge, "2.0"):
                model = torch.jit.trace(
                    model,
                    example_inputs=input_sample,
                    check_trace=False,
                    strict=jit_strict,
                    example_kwarg_inputs=example_kwarg_inputs)
            else:
                model = torch.jit.trace(
                    model, input_sample,
                    check_trace=False,
                    strict=jit_strict)
        except Exception:
            model = torch.jit.script(model)
    return model
