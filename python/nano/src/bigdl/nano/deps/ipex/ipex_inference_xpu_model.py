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
from bigdl.nano.utils.pytorch import apply_data_to_xpu, apply_data_to_half
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.utils.common import invalidInputError
import warnings


class PytorchIPEXPUModel(AcceleratedLightningModule):

    def __init__(self, model: torch.nn.Module, thread_num=None,
                 precision="fp32", use_ipex=False):
        super().__init__(model)
        invalidInputError(precision in ["fp32", "fp16"],
                          f"support precision for GPU device only for fp32 and fp16, but "
                          f"found {precision}")
        if not use_ipex:
            warnings.warn("IPEX is required to be installed for GPU device inferencing even "
                          "the model is not optimized by IPEX.")
        self.original_state_dict = None
        if use_ipex:
            self.original_state_dict = model.state_dict()
        if precision == "fp16":
            self.model = self.model.half()
        self.model = model.to('xpu')
        if use_ipex:
            self.model = ipex.optimize(self.model)
        self.thread_num = thread_num
        self.precision = precision
        self.use_ipex = use_ipex
        self._nano_context_manager = generate_context_manager(accelerator=None,
                                                              precision="fp32",
                                                              thread_num=thread_num,
                                                              enable_onednn=False,
                                                              use_xpu=True)

    def forward(self, *inputs, **kwargs):
        if self.precision == "fp16":
            inputs = tuple(map(lambda item: apply_data_to_half(item), inputs))
        inputs = tuple(map(lambda item: apply_data_to_xpu(item), inputs))
        for key, val in kwargs.items():
            new_val = apply_data_to_xpu(val)
            if self.precision == "fp16":
                new_val = apply_data_to_half(new_val)
            kwargs[key] = new_val
        return self.model(*inputs, **kwargs)

    def __getattr__(self, name: str):
        # the search order is:
        # 1. current instance, like channels_last will be found at this place
        # 2. super class, like model will be found at this place
        # 3. original model, like additional attributes of original model
        #    will be found at this place
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @property
    def status(self):
        status = super().status
        status.update({"thread_num": self.thread_num,
                       "checkpoint": "ckpt.pth",
                       "use_ipex": self.use_ipex,
                       "precision": self.precision,
                       "compression": "fp32"})
        return status

    def _load(path, model, input_sample=None, inplace=False):
        status = PytorchIPEXPUModel._load_status(path)
        checkpoint_path = path / status['checkpoint']
        state_dict = torch.load(checkpoint_path, map_location="xpu")
        model.eval()
        model.load_state_dict(state_dict)
        return PytorchIPEXPUModel(model, thread_num=status["thread_num"],
                                  precision=status["precision"],
                                  use_ipex=status["use_ipex"])

    def _save_model(self, path, compression="fp32"):
        invalidInputError(compression == "fp32", "Could not compress ckpt for XPU model.")
        if self.use_ipex:
            torch.save(self.original_state_dict, path / "ckpt.pth")
        else:
            torch.save(self.model.state_dict(), path / "ckpt.pth")
