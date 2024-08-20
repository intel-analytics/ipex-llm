
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

    def run_optimize_gpu_model(self, Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound):
        with torch.inference_mode():
            def pre_forward_hook(module, input, output, layer_name):
                self.pre_layer_outputs.append(output)

            def forward_hook(module, input, output, layer_name):
                self.layer_outputs.append(output)


            tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
            input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

            model = Model.from_pretrained(model_path,
                                        load_in_4bit=True,
                                        optimize_model=False,
                                        trust_remote_code=True)

            model = model.to(device)

            for layer_name, layer_module in model.named_modules():
                if layer_name == layer_before_RMSNorm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: pre_forward_hook(module, input,
                                                                                            output, layer_name))
                if layer_name == RMSNorm_layer:
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
                if layer_name == layer_before_RMSNorm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: replace_forward_hook(module, input,
                                                                                                output, layer_name))
                if layer_name == RMSNorm_layer:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_optimized_model = (opt_model(input_ids)).logits
            # the list `layer_output` has only one element.
            opt_layer_tensor = self.layer_outputs[0]
            opt_model.to('cpu')

            RMSNorm_output_diff = []
            for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
                if t1 is not None and t2 is not None:
                    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                        RMSNorm_output_diff.append(t1 - t2)
            max_diff_tensor = [torch.max(item).item() for item in RMSNorm_output_diff]
            print(max_diff_tensor)
            torch.xpu.empty_cache()
            del model
            del opt_model
            gc.collect()
            assert all(max_diff <= lower_bound for max_diff in max_diff_tensor)

    @pytest.mark.parametrize('Name, Model, Tokenizer, model_path',TEST_MODEL_LIST)
    def test_dynamic_functions(self, Name, Model, Tokenizer, model_path):
        if Name == "Llama2-7B":
            self.Llama2_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "ChatGLM2-6B":
            self.Chatglm2_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Mistral-7B-Instruct-v0.1":
            self.Mistral_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Baichuan2-7B-Chat":
            self.Baichuan_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Qwen-7B-Chat":
            self.Qwen_gpu_model(Name, Model, Tokenizer, model_path)

    def Llama2_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        layer_before_RMSNorm = "model.layers.30"
        RMSNorm_layer = "model.layers.31.input_layernorm"
        lower_bound = 2e-6
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound)

    def Chatglm2_gpu_model(self, Name, Model, Tokenizer, model_path):
        layer_before_RMSNorm = "transformer.encoder.layers.26"
        RMSNorm_layer = "transformer.encoder.layers.27.input_layernorm"
        lower_bound = 4e-6
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound)

    def Mistral_gpu_model(self, Name, Model, Tokenizer, model_path):
        layer_before_RMSNorm = "model.layers.30"
        RMSNorm_layer = "model.layers.31.input_layernorm"
        lower_bound = 2e-5
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound)

    def Baichuan_gpu_model(self, Name, Model, Tokenizer, model_path):
        layer_before_RMSNorm = "model.layers.30"
        RMSNorm_layer = "model.layers.31.input_layernorm"
        lower_bound = 1e-6
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound)

    def Qwen_gpu_model(self, Name, Model, Tokenizer, model_path):
        layer_before_RMSNorm = "transformer.h.30"
        RMSNorm_layer = "transformer.h.31.ln_1"
        lower_bound = 4e-6
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, RMSNorm_layer, layer_before_RMSNorm, lower_bound)
