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

import os
import pytest

import torch
from transformers import LlamaTokenizer, AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel


device = os.environ['DEVICE']
print(f'Running on {device}')
if device == 'xpu':
    import intel_extension_for_pytorch as ipex

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"


@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('FALCON_7B_ORIGIN_PATH'))
])

def test_optimize_model(Model, Tokenizer, model_path):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=False,
                                trust_remote_code=True)
    model = model.to(device)
    logits_base_model = (model(input_ids)).logits
    model.to('cpu')  # deallocate gpu memory

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=True,
                                trust_remote_code=True)
    model = model.to(device)
    logits_optimized_model = (model(input_ids)).logits
    model.to('cpu')
    
    diff = abs(logits_base_model - logits_optimized_model).flatten()

    assert any(diff) is False


class Test_Optimize_Gpu_Model:
    def setup(self):
        self.lower_bound = 3e-3
        
        self.layer_inputs = []
        self.layer_outputs = []

    def run_optimize_gpu_model(self, Model, Tokenizer, model_path, self_attn):
        def forward_hook(module, output, layer_name):
            self.layer_outputs.append(output)

        def pre_hook(module, input):
            self.layer_inputs.append(input)

        def load_pre_hook(module, input):
            return self.layer_inputs[0]

        tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        model = Model.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      optimize_model=False,
                                      trust_remote_code=True)
        model = model.to(device)

        for layer_name, layer_module in model.named_modules():
            if layer_name == self_attn:
                layer_module.register_forward_pre_hook(
                    lambda module, input: pre_hook(module, input))
                layer_module.register_forward_hook(
                    lambda module, output, layer_name=layer_name: forward_hook(module, 
                                                                               output, layer_name))
        logits_base_model = (model(input_ids)).logits
        # the list `layer_output` has only one element.
        layer_tensor = self.layer_outputs.pop()

        del model

        opt_model = Model.from_pretrained(model_path,
                                          load_in_4bit=True,
                                          optimize_model=True,
                                          trust_remote_code=True)
        opt_model = opt_model.to(device)

        for layer_name, layer_module in opt_model.named_modules():
            if layer_name == self_attn:
                layer_module.register_forward_pre_hook(
                    lambda module, input: load_pre_hook(module, input))
                layer_module.register_forward_hook(
                    lambda module, output, layer_name=layer_name: forward_hook(module, 
                                                                               output, layer_name))
        logits_optimized_model = (opt_model(input_ids)).logits
        # the list `layer_output` has only one element.
        opt_layer_tensor = self.layer_outputs.pop()

        del opt_model

        attn_output_diff = []
        for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
            if t1 is not None and t2 is not None:
                if not isinstance(t1, tuple) and not isinstance(t2, tuple):
                    attn_output_diff.append(t1 - t2)
                else:
                    for i, (t3, t4) in enumerate(zip(t1, t2)):
                        attn_output_diff.append(t3 - t4)

        max_diff_tensor = [torch.max(item).item() for item in attn_output_diff]
        assert all(max_diff <= self.lower_bound for max_diff in max_diff_tensor)


    def test_falcon_gpu_model(self):

        Model = AutoModelForCausalLM
        Tokenizer = AutoTokenizer
        model_path = os.environ.get('FALCON_7B_ORIGIN_PATH')
        # currently only need to compare the output of one self-attention layer.
        self_attn = "transformer.h.31.self_attention"

        self.run_optimize_gpu_model(Model, Tokenizer, model_path, self_attn)


    def test_llama_gpu_model(self):

        Model = AutoModelForCausalLM
        Tokenizer = AutoTokenizer
        model_path = os.environ.get('LLAMA_ORIGIN_PATH')
        # currently only need to compare the output of one self-attention layer.
        self_attn = "model.layers.31.self_attn"

        self.run_optimize_gpu_model(Model, Tokenizer, model_path, self_attn)


if __name__ == '__main__':
    pytest.main([__file__])
