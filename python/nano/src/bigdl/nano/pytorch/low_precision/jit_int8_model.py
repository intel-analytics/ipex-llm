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


from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.pytorch import jit_convert
from bigdl.nano.utils.common import compare_version
import torch
from torch.ao.quantization import QConfig, get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import operator
from collections.abc import Sequence


class PytorchJITINT8Model(AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, calib_data, q_config=None,
                 input_sample=None, channels_last=False, thread_num=None,
                 from_load=False, jit_strict=True, jit_method=None,
                 enable_onednn=False, example_kwarg_inputs=None):
        '''
        This is the accelerated model for pytorch fx quantization and jit.
        All the external API is based on InferenceOptimizer, so what we have here is
        basically internal APIs and subject to change.

        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param calib_data: calibration data is required for static quantization.
        :param q_config: We support 2 types of customized quantization config:

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
               otherwise will be ignored.
        '''
        super().__init__(model)

        enable_onednn = False
        # TODO: since onednn cooperates well with other nano methods, it is set to True
        # by default in InferenceOptimizer.quantize(). However, it will lead to strange
        # error in fx quantization during inference. Therefore, we disable it by hand.
        # Actually, prepare_fx supports the onednn backend, related design is in here:
        # https://github.com/pytorch/pytorch/pull/69820
        #
        # To use it, we can use below statement to replace the fbgemm(default) backend
        # in line 104.
        # qconfig_mapping = get_default_qconfig_mapping("onednn")
        #
        # A simple test of performance is shown in
        # https://github.com/intel-analytics/BigDL/pull/7483

        if from_load:
            self.channels_last = channels_last
            self.jit_strict = jit_strict
            self.jit_method = jit_method
            self.enable_onednn = enable_onednn
            self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                                  precision="int8",
                                                                  thread_num=thread_num,
                                                                  enable_onednn=enable_onednn)
            return

        self.original_state_dict = model.state_dict()
        self.channels_last = channels_last
        self.jit_strict = jit_strict
        self.jit_method = jit_method
        self.enable_onednn = enable_onednn
        self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                              precision="int8",
                                                              thread_num=thread_num,
                                                              enable_onednn=enable_onednn)
        self.thread_num = thread_num
        self.original_model = model
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if q_config is None:
            self.q_config = get_default_qconfig_mapping("fbgemm")
        else:
            if isinstance(q_config, QConfig):
                self.q_config = {'': q_config}
            else:
                self.q_config = q_config

        if input_sample is None:
            input_sample = next(iter(calib_data))
            if isinstance(input_sample, (tuple, list)) and len(input_sample) > 1:
                input_sample = input_sample[0]
                if self.channels_last:
                    if isinstance(input_sample, torch.Tensor):
                        input_sample = input_sample.to(memory_format=torch.channels_last)
                    else:
                        input_sample = tuple(map(lambda x: x.to(memory_format=torch.channels_last),
                                                 input_sample))

        self.model = prepare_fx(self.model, self.q_config,
                                example_inputs=(input_sample,))

        # TODO: multiple input data not supported during calibration
        # the same problem as ipex_quantization model
        for x in calib_data:
            if isinstance(x, (tuple, list)) and len(x) > 1:
                x = x[0]
            if isinstance(x, Sequence):
                self.model(*x)
            else:
                self.model(x)

        self.model = convert_fx(self.model)

        with torch.no_grad():
            if example_kwarg_inputs is not None:
                input_sample = None
            self.model = jit_convert(self.model, input_sample,
                                     jit_method=jit_method,
                                     jit_strict=jit_strict,
                                     example_kwarg_inputs=example_kwarg_inputs)
            self.model = torch.jit.freeze(self.model)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        if self.channels_last:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    def __getattr__(self, name: str):
        # the search order is:
        # 1. current instance, like channels_last will be found at this place
        # 2. super class, like model will be found at this place
        # 3. original model, like additional attributes of original model
        #    will be found at this place
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_model, name)

    @property
    def status(self):
        status = super().status
        status.update({"channels_last": self.channels_last,
                       "checkpoint": "ckpt.pth",
                       "thread_num": self.thread_num,
                       "jit_strict": self.jit_strict,
                       "jit_method": self.jit_method,
                       "enable_onednn": self.enable_onednn})
        return status

    @staticmethod
    def _load(path):
        status = PytorchJITINT8Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model = torch.jit.freeze(model)
        from_load = True
        thread_num = None
        if status["thread_num"] is not None and status['thread_num'] != {}:
            thread_num = int(status['thread_num'])
        return PytorchJITINT8Model(model,
                                   calib_data=None,
                                   channels_last=status.get('channels_last', False),
                                   from_load=from_load,
                                   thread_num=thread_num,
                                   jit_strict=status.get('jit_strict', True),
                                   jit_method=status.get('jit_method', None),
                                   enable_onednn=status.get('enable_onednn', False))

    def _save_model(self, path, compression="fp32"):
        torch.jit.save(self.model, path / "ckpt.pth")
