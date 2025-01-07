
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
import gc
import pytest

import torch
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

device = os.environ['DEVICE']
print(f'Running on {device}')

PROMPT = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
TEST_MODEL_LIST = [
    ("MPT-7B", AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    ("Llama2-7B", AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    ("ChatGLM2-6B", AutoModel, AutoTokenizer, os.environ.get('CHATGLM2_6B_ORIGIN_PATH')),
    ("Mistral-7B-Instruct-v0.1", AutoModelForCausalLM, AutoTokenizer, os.environ.get('MISTRAL_7B_INSTRUCT_V0_1_ORIGIN_PATH')),
    ("Baichuan2-7B-Chat", AutoModelForCausalLM, AutoTokenizer, os.environ.get('BAICHUAN2_7B_ORIGIN_PATH')),
    # ("Qwen-7B-Chat", AutoModelForCausalLM, AutoTokenizer, os.environ.get('QWEN_7B_ORIGIN_PATH')), # qwen requires transformers<4.37.0
]

class Test_Optimize_Gpu_Model:
    def setup_method(self):

        self.layer_outputs = []
        self.pre_layer_outputs = []

    def run_optimize_gpu_model(self, Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound):
        with torch.inference_mode():
            def forward_hook(module, input, output, layer_name):
                self.layer_outputs.append(output)

            def pre_forward_hook(module, input, output, layer_name):
                self.pre_layer_outputs.append(output)

            tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
            input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

            model = Model.from_pretrained(model_path,
                                        load_in_4bit=True,
                                        optimize_model=False,
                                        trust_remote_code=True)
            model = model.to(device)
            for layer_name, layer_module in model.named_modules():
                if layer_name == layer_norm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: pre_forward_hook(module, input,
                                                                                            output, layer_name))
                if layer_name == self_attn:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_base_model = (model(input_ids)).logits
            # the list `layer_output` has only one element.
            layer_tensor = self.layer_outputs.pop()
            model.to('cpu')

            opt_model = Model.from_pretrained(model_path,
                                            load_in_4bit=True,
                                            optimize_model=True,
                                            trust_remote_code=True)
            opt_model = opt_model.to(device)


            def replace_forward_hook(module, input, output, layer_name):
                output = self.pre_layer_outputs[0]
                return output

            for layer_name, layer_module in opt_model.named_modules():
                if layer_name == layer_norm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: replace_forward_hook(module, input,
                                                                                                output, layer_name))
                if layer_name == self_attn:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_optimized_model = (opt_model(input_ids)).logits
            # the list `layer_output` has only one element.
            opt_layer_tensor = self.layer_outputs[0]
            opt_model.to('cpu')

            attn_output_diff = []
            for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
                if t1 is not None and t2 is not None:
                    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                        # 'attn_output' is of type torch.Tensor.
                        attn_output_diff.append(t1 - t2)
                    elif isinstance(t1, tuple) and isinstance(t2, tuple):
                        # if 'past_key_value'is of type tuple
                        for i, (t3, t4) in enumerate(zip(t1, t2)):
                            attn_output_diff.append(t3 - t4)
                    else:
                        # if 'past_key_value'is of type Cache, get last layer cache pair (key, value)
                        attn_output_diff.append(t1[-1][0] - t2[-1][0])
                        attn_output_diff.append(t1[-1][1] - t2[-1][1])

            max_diff_tensor = [torch.max(item).item() for item in attn_output_diff]
            print(max_diff_tensor)
            torch.xpu.empty_cache()
            del model
            del opt_model
            gc.collect()

            assert all(max_diff <= lower_bound for max_diff in max_diff_tensor)

    @pytest.mark.parametrize('Name, Model, Tokenizer, model_path',TEST_MODEL_LIST)
    def test_dynamic_functions(self, Name, Model, Tokenizer, model_path):
        if Name == "MPT-7B":
            self.MPT_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Llama2-7B":
            self.Llama2_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "ChatGLM2-6B":
            self.Chatglm2_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Mistral-7B-Instruct-v0.1":
            self.Mistral_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Baichuan2-7B-Chat":
            self.Baichuan_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Qwen-7B-Chat":
            self.Qwen_gpu_model(Name, Model, Tokenizer, model_path)


    def MPT_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.blocks.31.norm_1"
        self_attn = "transformer.blocks.31.attn"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Llama2_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last self-attention layer.
        layer_norm = "model.layers.31.input_layernorm"
        self_attn = "model.layers.31.self_attn"
        lower_bound = 2e-1
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Chatglm2_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.encoder.layers.27.input_layernorm"
        self_attn = "transformer.encoder.layers.27.self_attention"
        lower_bound = 1e-1
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Mistral_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "model.layers.31.input_layernorm"
        self_attn = "model.layers.31.self_attn"
        if os.environ['RUNNER_OS'] == "Windows":
            lower_bound = 2e-2
        else:
            lower_bound = 2e-2
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Baichuan_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "model.layers.31.input_layernorm"
        self_attn = "model.layers.31.self_attn"
        lower_bound = 2e-2
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Qwen_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.h.31.ln_1"
        self_attn = "transformer.h.31.attn"
        lower_bound = 2e-2
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)
