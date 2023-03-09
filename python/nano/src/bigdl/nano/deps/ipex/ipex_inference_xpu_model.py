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

from bigdl.nano.pytorch.model import AcceleratedLightningModule
import torch
import intel_extension_for_pytorch as ipex
from bigdl.nano.utils.pytorch import apply_data_to_xpu


class PytorchIPEXPUModel(AcceleratedLightningModule):

    def __init__(self, model: torch.nn.Module, thread_num=None):
        self.model = model.to('xpu')
        self.thread_num = thread_num
        self._nano_context_manager = generate_context_manager(accelerator=None,
                                                              precision="fp32",
                                                              thread_num=thread_num,
                                                              enable_onednn=False)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        # move the data to xpu
        inputs = tuple(map(lambda item: apply_data_to_xpu(item), inputs))

        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs
    
    @property
    def status(self):
        status = super().status
        status.update({"thread_num": self.thread_num,
                       "checkpoint": "ckpt.pth"})
        return status

    def _load(path, model, input_sample=None, inplace=False):
        status = PytorchIPEXPUModel._load_status(path)
        checkpoint_path = path / status['checkpoint']
        state_dict = torch.load(checkpoint_path)
        model.eval()
        model.load_state_dict(state_dict)
        return PytorchIPEXPUModel(model, thread_num=status["thread_num"])
