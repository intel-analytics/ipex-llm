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
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer
 
device = os.environ['DEVICE']
print(f'Running on {device}')
 
PROMPT = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
TEST_MODEL_LIST = [
    ("Qwen-7B-Chat", AutoModelForCausalLM, AutoTokenizer, os.environ.get('QWEN_7B_ORIGIN_PATH')),
    ("Mistral-7B-Instruct-v0.1", AutoModelForCausalLM, AutoTokenizer, os.environ.get('MISTRAL_7B_INSTRUCT_V0_1_ORIGIN_PATH')),
    ("Llama2-7B", AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    ("Falcon-7B", AutoModelForCausalLM, AutoTokenizer, os.environ.get('FALCON_7B_ORIGIN_PATH'))
]
 
class Test_Optimize_Gpu_Model:
    def setup_method(self):
        self.layer_outputs = []
        self.pre_layer_outputs = []
 
    def run_optimize_gpu_model(self, Name, Model, Tokenizer, model_path, test_layer, layer_before_test_layer, lower_bound):
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
                if layer_name == layer_before_test_layer:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: pre_forward_hook(module, input,
                                                                                            output, layer_name))
                if layer_name == test_layer:
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
                if layer_name == layer_before_test_layer:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: replace_forward_hook(module, input,
                                                                                                output, layer_name))
                if layer_name == test_layer:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_optimized_model = (opt_model(input_ids)).logits
            # the list `layer_output` has only one element.
            opt_layer_tensor = self.layer_outputs[0]
            opt_model.to('cpu')
 
 
            test_layer_output_diff = []
            for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
                if t1 is not None and t2 is not None:
                    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                        test_layer_output_diff.append(t1 - t2)
                    else:
                        # 'past_key_value'is of type tuple as default.
                        for i, (t3, t4) in enumerate(zip(t1, t2)):
                            test_layer_output_diff.append(t3 - t4)
 
            max_diff_tensor = [torch.max(item).item() for item in test_layer_output_diff]
            print(max_diff_tensor)
            torch.xpu.empty_cache()
            del model
            del opt_model
            gc.collect()
            assert all(max_diff <= lower_bound for max_diff in max_diff_tensor)
   
    @pytest.mark.parametrize('Name, Model, Tokenizer, model_path',TEST_MODEL_LIST)
    # Qwen-7B: Testing MLP layer
    # Mistral-7B: Testing MLP layer
    # Llama2: Testing Decoder Layer
    # Falcon-7B: Testing LayerNorm layer
    def test_dynamic_functions(self, Name, Model, Tokenizer, model_path):
        if Name == "Qwen-7B-Chat":
            self.Qwen_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Mistral-7B-Instruct-v0.1":
            self.Mistral_7B_Instruct_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Llama2-7B":
            self.Llama2_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Falcon-7B":
            self.Falcon_7B_gpu_model(Name, Model, Tokenizer, model_path)

 
    def Qwen_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last mlp layer.
        layer_before_MLP = "transformer.h.31.ln_2"
        MLP_layer = "transformer.h.31.mlp"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, MLP_layer, layer_before_MLP, lower_bound)
   
    def Mistral_7B_Instruct_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last mlp layer.
        layer_before_MLP = "model.layers.31.post_attention_layernorm"
        MLP_layer = "model.layers.31.mlp"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, MLP_layer, layer_before_MLP, lower_bound)

    def Llama2_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last Decoder layer.
        layer_before_Decoder = "model.layers.30"
        Decoder_layer = "model.layers.31"
        lower_bound = 5e-2
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, Decoder_layer, layer_before_Decoder, lower_bound)
    
    def Falcon_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last LayerNorm layer.
        layer_before_LayerNorm = "transformer.h.30"
        LayerNorm_layer = "transformer.h.31.input_layernorm"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, LayerNorm_layer, layer_before_LayerNorm, lower_bound)