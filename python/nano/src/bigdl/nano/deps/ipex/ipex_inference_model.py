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


class IPEXJITModel:
    def __init__(self,
                 model,
                 input_sample=None,
                 use_ipex=False,
                 use_jit=False,
                 channels_last=None,
                 from_load=False):
        if from_load:
            self.model = model
            self.use_ipex = use_ipex
            self.use_jit = use_jit
            self.channels_last = channels_last
            return
        self.channels_last = use_ipex if (channels_last is None or not use_ipex) else channels_last
        model.eval()
        self.original_state_dict = model.state_dict()
        self.model = model
        self.use_ipex = use_ipex
        self.use_jit = use_jit
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.use_ipex:
            self.model = ipex.optimize(self.model)
        if self.use_jit:
            self.model = torch.jit.trace(self.model, input_sample)
            self.model = torch.jit.freeze(self.model)

    def forward_step(self, *inputs):
        if self.channels_last:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def _save_model(self, path):
        if self.use_jit:
            self.model.save(path / "ckpt.pth")
        else:
            torch.save(self.original_state_dict, path / "ckpt.pth")


class PytorchIPEXJITModel(IPEXJITModel, AcceleratedLightningModule):
    def __init__(self, model, input_sample=None, use_ipex=False,
                 use_jit=False, channels_last=None, from_load=False):
        AcceleratedLightningModule.__init__(self, None)
        IPEXJITModel.__init__(self, model, input_sample=input_sample,
                              use_ipex=use_ipex, use_jit=use_jit, from_load=from_load)

    def on_forward_start(self, inputs):
        return inputs

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
        else:
            state_dict = torch.load(checkpoint_path)
            model.eval()
            model.load_state_dict(state_dict)
            if status["channels_last"]:
                model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model)
        return PytorchIPEXJITModel(model, use_ipex=status['use_ipex'],
                                   use_jit=status['use_jit'],
                                   channels_last=status['channels_last'],
                                   from_load=True)

    def _save_model(self, path):
        super()._save_model(path)
