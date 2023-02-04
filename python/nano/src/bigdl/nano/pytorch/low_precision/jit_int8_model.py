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
from bigdl.nano.pytorch.context_manager import generate_context_manager

import torch
import torch.ao.quantization
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from collections.abc import Sequence


class PytorchJITINT8Model(AcceleratedLightningModule):
    def __init__(self, model: torch.nn.Module, calib_data, q_config=None,
                 input_sample=None, channels_last=None, thread_num=None,
                 from_load=False, jit_strict=True, jit_method=None):
        super().__init__(model)
        if from_load:
            self.jit_strict = jit_strict
            self.jit_method = jit_method
            self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                                  precision="int8",
                                                                  thread_num=thread_num)
            return
        
        self.original_state_dict = model.state_dict()
        self.jit_strict = jit_strict
        self.jit_method = jit_method
        self._nano_context_manager = generate_context_manager(accelerator="jit",
                                                              precision="int8",
                                                              thread_num=thread_num)
        self.thread_num = thread_num
        self.original_model = model

        if q_config is None:
            self.q_config = torch.ao.quantization.default_qconfig
        else:
            self.q_config = q_config
        
        if input_sample is None:
            input_sample = next(iter(calib_data))
            if isinstance(input_sample, (tuple, list)) and len(input_sample) > 1:
                input_sample = input_sample[0]

        self.model = prepare_fx(self.model, {'' : self.q_config},
                                example_inputs=input_sample)

        for x in calib_data:
            if isinstance(x, (tuple, list)) and len(x) > 1:
                x = x[0]
            if isinstance(x, Sequence):
                self.model(*x)
            else:
                self.model(x)

        self.model = convert_fx(self.model)

        with torch.no_grad():
            if self.jit_method == 'trace':
                self.model = torch.jit.trace(self.model, input_sample,
                                             strict=jit_strict)
            elif self.jit_method == 'script':
                self.model = torch.jit.script(self.model)
            else:
                try:
                    self.model = torch.jit.trace(self.model, input_sample,
                                                 strict=jit_strict)
                except Exception:
                    self.model = torch.jit.script(self.model)
            self.model = torch.jit.freeze(self.model)
            
    @property
    def forward_args(self):
        return [input_value.debugName() for input_value in self.model.graph.inputs()
                if not input_value.debugName().startswith('self')]

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
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
        status.update({"checkpoint": "ckpt.pth",
                       "thread_num": self.thread_num,
                       "jit_strict": self.jit_strict,
                       'jit_method': self.jit_method})
        return status

    @staticmethod
    def _load(path, model):
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
                              from_load=from_load,
                              thread_num=thread_num,
                              jit_strict=status["jit_strict"],
                              jit_method=status["jit_method"])

    def _save_model(self, path):
        self.model.save(path / "ckpt.pth")
