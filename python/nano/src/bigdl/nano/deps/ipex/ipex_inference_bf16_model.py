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


from .ipex_inference_model import PytorchIPEXJITModel
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.utils.common import _avx512_checker
from bigdl.nano.utils.common import invalidInputError
import torch


class PytorchIPEXJITBF16Model(PytorchIPEXJITModel):
    def __init__(self, model, input_sample=None, use_ipex=False,
                 use_jit=False, channels_last=None, channels_last_available=[],
                 thread_num=None, from_load=False, inplace=False, jit_strict=True,
                 jit_method=None, weights_prepack=None, enable_onednn=True):
        '''
        This is the accelerated model for pytorch and ipex/jit.
        All the external API is based on InferenceOptimizer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchIPEXJITBF16Model will serve for bf16 and ipex>1.9 models.
        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param input_sample: torch tensor indicate the data sample to be used
               for tracing.
        :param use_ipex: if use ipex to optimize the model
        :param use_jit: if use jit to accelerate the model
        :param channels_last: if set model and data to be channels-last mode.
               the parameter will be ignored if use_ipex is False.
        :param thread_num: the thread num allocated for this model.
        :param from_load: this will only be set by _load method.
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
        if use_ipex:
            invalidInputError(
                self._check_cpu_isa,
                errMsg="Applying IPEX BF16 optimization needs the cpu support avx512.",
                fixMsg="Please set use_ipex to False or not set precision to bf16."
            )

        PytorchIPEXJITModel.__init__(self, model, input_sample=input_sample, use_ipex=use_ipex,
                                     dtype=torch.bfloat16, use_jit=use_jit,
                                     channels_last=channels_last,
                                     channels_last_available=channels_last_available,
                                     from_load=from_load, inplace=inplace, jit_strict=jit_strict,
                                     jit_method=jit_method, weights_prepack=weights_prepack)
        _accelerator = "jit" if use_jit is True else None
        self._nano_context_manager = generate_context_manager(accelerator=_accelerator,
                                                              precision="bf16",
                                                              thread_num=thread_num,
                                                              enable_onednn=enable_onednn)

    @property
    def _check_cpu_isa(self):
        """Indicator to verify if cpu supports avx512"""
        return _avx512_checker()

    @property
    def status(self):
        status = super().status
        status.update({"precision": "bfloat16"})
        return status

    @staticmethod
    def _load(path, model, inplace=False):
        status = PytorchIPEXJITBF16Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        if status["use_jit"]:
            if status["use_ipex"]:
                import intel_extension_for_pytorch as ipex
            model = torch.jit.load(checkpoint_path)
            model.eval()
            if status["use_ipex"]:
                model = torch.jit.freeze(model)
            from_load = True
        else:
            state_dict = torch.load(checkpoint_path)
            model.eval()
            model.load_state_dict(state_dict)
            from_load = False
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return PytorchIPEXJITBF16Model(model, use_ipex=status['use_ipex'],
                                       use_jit=status['use_jit'],
                                       channels_last=status['channels_last'],
                                       channels_last_available=status['channels_last_available'],
                                       from_load=from_load,
                                       thread_num=thread_num,
                                       inplace=inplace,
                                       jit_strict=status.get('jit_strict', True),
                                       jit_method=status.get('jit_method', None),
                                       weights_prepack=status.get('weights_prepack', None),
                                       enable_onednn=status.get('enable_onednn', True))
