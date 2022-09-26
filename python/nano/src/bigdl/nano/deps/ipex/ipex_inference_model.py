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

from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
import intel_extension_for_pytorch as ipex
import torch


class PytorchIPEXJITModel(AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, input_sample=None, use_ipex=False, dtype=None,
                 use_jit=False, channels_last=None, from_load=False):
        """
        This is the accelerated model for pytorch and ipex/jit.
        All the external API is based on Trainer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchIPEXJITModel will serve for fp32 and ipex>1.9 models.
        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param input_sample: torch tensor indicate the data sample to be used
               for tracing.
        :param use_ipex: if use ipex to optimize the model
        :param dtype (torch.dtype): Only works for ``torch.bfloat16``. Model parameters
                                    will be casted to ``torch.bfloat16`` if dtype is set
                                    to ``torch.bfloat16``. The default value is None,
                                    meaning do nothing.
        :param use_jit: if use jit to accelerate the model
        :param channels_last: if set model and data to be channels-last mode.
               the parameter will be ignored if use_ipex is False.
        :param from_load: this will only be set by _load method.
        """
        model.eval()
        super().__init__(model)
        if from_load:
            self.use_ipex = use_ipex
            self.use_jit = use_jit
            self.channels_last = channels_last
            return
        self.channels_last = use_ipex if (channels_last is None or not use_ipex) else channels_last
        self.original_state_dict = model.state_dict()
        self.use_ipex = use_ipex
        self.use_jit = use_jit
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.use_ipex:
            self.model = ipex.optimize(self.model, dtype=dtype)
        if self.use_jit:
            self.model = torch.jit.trace(self.model, input_sample)
            self.model = torch.jit.freeze(self.model)

    @property
    def forward_args(self):
        return [input_value.debugName() for input_value in self.model.graph.inputs()
                if not input_value.debugName().startswith('self')]

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        if self.channels_last:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"use_ipex": self.use_ipex,
                       "use_jit": self.use_jit,
                       "channels_last": self.channels_last,
                       "checkpoint": "ckpt.pth"})
        return status

    @staticmethod
    def _load(path, model):
        status = PytorchIPEXJITModel._load_status(path)
        checkpoint_path = path / status['checkpoint']
        if status["use_jit"]:
            model = torch.jit.load(checkpoint_path)
            model.eval()
            model = torch.jit.freeze(model)
            from_load = True
        else:
            state_dict = torch.load(checkpoint_path)
            model.eval()
            model.load_state_dict(state_dict)
            from_load = False
        return PytorchIPEXJITModel(model, use_ipex=status['use_ipex'],
                                   use_jit=status['use_jit'],
                                   channels_last=status['channels_last'],
                                   from_load=from_load)

    def _save_model(self, path):
        if self.use_jit:
            self.model.save(path / "ckpt.pth")
        else:
            torch.save(self.original_state_dict, path / "ckpt.pth")
