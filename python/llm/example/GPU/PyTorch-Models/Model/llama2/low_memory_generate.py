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
# The code is adapted from: https://www.kaggle.com/code/simjeg/platypus2-70b-without-wikipedia-rag.
#

import argparse
import gc
import os
import time
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ipex_llm import optimize_model
from ipex_llm.transformers.low_bit_linear import FP4Params, LowBitLinear

MAX_LENGTH = 4096
# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
LLAMA2_PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""


# Modified based on https://github.com/huggingface/accelerate/blob/d25efa71ce76a5f5911a1fc6c039979d7248596f/src/accelerate/utils/modeling.py#L238
def set_module_tensor_to_device_with_cache(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor],
    dtype: Optional[Union[str, torch.dtype]] = None,
    cache_dict: Optional[Dict[str, FP4Params]] = None,
    max_cache_num: Optional[int] = 100,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        param_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
        cache_dict (`Dict`, *optional*):
            The cache dict to save layer weights. This can improve the loading speed.
        max_cache_num (`int`, *optional*):
            The maximum number of weights saved in the cache_dict. You can adjust this number based on your GPU memory.
            Default is 100.
    """
    original_tensor_name = tensor_name
    assert value is not None

    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    # Use cache to load weights
    if original_tensor_name in cache_dict:
        module._parameters[tensor_name] = cache_dict[original_tensor_name]
        return

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    if dtype is None:
        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
        value = value.to(old_value.dtype)
    elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        value = value.to(dtype)

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    with torch.no_grad():
        if isinstance(module, LowBitLinear):
            # load cpu int4 weights
            new_value = FP4Params(data=value,
                                  requires_grad=False,
                                  quantized=True,
                                  _shape=(module.out_features, module.in_features),
                                  convert_shape_only=False,
                                  qtype=2).to(device)
            if len(cache_dict) < max_cache_num:
                cache_dict.update({original_tensor_name: new_value})
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif isinstance(module, LowBitLinear):
            module._parameters[tensor_name] = new_value
        elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
            param_cls = type(module._parameters[tensor_name])
            new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value
        else:
            raise NotImplementedError


class LowMemoryLlama(GenerationMixin):
    def __init__(self, model_path: str, splitted_weights_path: str, max_cache_num: Optional[int] = 100):
        """
        Low memory version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.

        Parameters
        ----------
        model_path (`str`):
            The huggingface repo id or path to the huggingface checkpoint folder that including config.json and tokenizer.model.
        splitted_weights_path (`str`):
            The folder including int4 weights per layer. You can use `LowMemoryLlama.split_and_convert_to_cpu_int4_weights` to generate those weights.
        max_cache_num (`int`, *optional*):
            The maximum number of weights saved in the cache_dict. You can adjust this number based on your GPU memory.
            Default is 100.
        """
        super().__init__()

        # Save parameters
        self.model_path = model_path
        self.splitted_weights_path = splitted_weights_path
        self.device = torch.device('xpu:0')
        self.layer_weight_cache = {} # initialize weight cache dict
        self.max_cache_num = max_cache_num

        # Create model
        self._create_model()
        # Check if `self.splitted_weights_path` exists
        self._check_split_weight_path()

        # Initialize attention mask and position ids to further improve the inference speed
        self._generate_att_mask_and_pos_id()

    @classmethod
    def split_and_convert_to_cpu_int4_weights(cls, model_path, safetensor_per_layer_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)
        model = optimize_model(model)

        state_dict = model.state_dict()
        layer_names = ["model.embed_tokens."] + [f"model.layers.{i}." for i in range(len(model.model.layers))] + ["model.norm.", "lm_head."]
        for layer_name in tqdm(layer_names):
            local_state_dict = {k: v.contiguous() for k, v in state_dict.items() if k.startswith(layer_name)}
            save_name = os.path.join(safetensor_per_layer_path, f"{layer_name}safetensors")
            save_file(local_state_dict, save_name)
        print(f'Save splitted safetensor weights to {safetensor_per_layer_path}')

    def _create_model(self):
        # Load config
        self.config = AutoConfig.from_pretrained(self.model_path)
        if self.config.pretraining_tp > 1:
            warnings.warn("Set config.pretraining_tp = 1 to use int4 inference")
            self.config.pretraining_tp = 1
        if not self.config.use_cache:
            warnings.warn("Set config.use_cache to further improve performance")
            self.config.use_cache = True

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Model initialization
        self._init_model()
        # Tie with self.model
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(len(self.model.model.layers))] + ["model.norm", "lm_head"]
        self.generation_config = self.model.generation_config
        self.main_input_name = self.model.main_input_name

    def _check_split_weight_path(self):
        for layer_name in self.layer_names:
            split_weight_path = os.path.join(self.splitted_weights_path,  f"{layer_name}.safetensors")
            if not os.path.exists(split_weight_path):
                raise FileNotFoundError(f"Weight file {split_weight_path} is missing."
                                        f"You can generate it using `LowMemoryLlama.split_and_convert_to_cpu_int4_weights`.")

    def _generate_att_mask_and_pos_id(self):
        self.attention_mask = torch.full((MAX_LENGTH, MAX_LENGTH), torch.finfo(torch.float16).min, device=self.device)
        mask_cond = torch.arange(self.attention_mask.size(-1), device=self.device)
        self.attention_mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.attention_mask.size(-1), 1), 0)
        self.attention_mask = self.attention_mask.to(torch.float16)[None, None, :, :]
        self.position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[None, :]

    def _init_model(self):
        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
            self.model = optimize_model(self.model)
            self.model.tie_weights()

        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) \
                        + [self.model.model.norm, self.model.lm_head]

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device_with_cache(self.model, buffer_name, self.device, value=buffer, dtype=buffer.dtype,
                                                   cache_dict=self.layer_weight_cache, max_cache_num=self.max_cache_num)

    def move_layer_to_device(self, state_dict):
        for param_name, param in state_dict.items():
            set_module_tensor_to_device_with_cache(self.model, param_name, self.device, value=param, dtype=param.dtype,
                                                   cache_dict=self.layer_weight_cache, max_cache_num=self.max_cache_num)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,
                input_ids: torch.LongTensor = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                *args,
                **kwargs,
        ):
        from packaging import version
        trans_version = transformers.__version__
        if version.parse(trans_version) >= version.parse("4.36.0"):
            transformers_4_36 = True
        else:
            transformers_4_36 = False

        # Reinit model and clean memory
        del self.model
        gc.collect()
        torch.xpu.empty_cache()
        self._init_model()

        # Send batch to device
        inputs = input_ids.to(self.device)

        current_shape = inputs.shape[1]
        # Set up kv cache
        if transformers_4_36:
            from transformers.cache_utils import Cache, DynamicCache
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            pre_shape = past_key_values.get_usable_length(current_shape)
        else:
            if past_key_values is not None:
                pre_shape = past_key_values[0][0].size(2)
            else:
                pre_shape = 0
                past_key_values = [None] * len(self.model.model.layers)

        with torch.inference_mode():
            # Generate attention mask and position ids
            pos = self.position_ids[:, pre_shape : current_shape + pre_shape]
            attn = self.attention_mask[:, :, -current_shape:, - current_shape - pre_shape:]

            for (layer_name, layer) in tqdm(zip(self.layer_names, self.layers), total=len(self.layers)):

                # Load current layer to device
                state_dict = load_file(os.path.join(self.splitted_weights_path,  f"{layer_name}.safetensors"), device="cpu")
                self.move_layer_to_device(state_dict)

                # Run layer
                if layer_name in ("model.embed_tokens", "model.norm", "lm_head"):
                    inputs = layer(inputs)
                else:
                    decoder_layer_index = int(layer_name.split('.')[-1])
                    past_key_value = past_key_values if transformers_4_36 else past_key_values[decoder_layer_index]
                    inputs, new_kv_cache = layer(inputs, use_cache=True, past_key_value=past_key_value,
                                                 position_ids=pos, attention_mask=attn)
                    if transformers_4_36:
                        past_key_values = new_kv_cache
                    else:
                        past_key_values[decoder_layer_index] = new_kv_cache

                # Delete weight before moving to('meta')
                for module in layer.modules():
                    if hasattr(module, "weight"):
                        del module.weight

                # Remove previous layer from memory (including buffers)
                layer.to("meta")

        result = CausalLMOutputWithPast(
            logits=inputs.detach(),
            past_key_values=past_key_values,
        )
        return result

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`,'
                             ' `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--splitted-weights-path', type=str, required=True,
                        help='The folder saving per-layer weights. You can use'
                         ' LowMemoryLlama.split_and_convert_to_cpu_int4_weights() to generate those weights.')
    parser.add_argument('--split-weight', action='store_true',
                        help='Whether to split weights by layer. If this argument is enabled, per-layer weights will'
                             ' be generated and saved to `--splitted-weights-path`. This argument only needs to be'
                             ' enabled once for the same model.')
    parser.add_argument('--max-cache-num', type=int, default=200,
                        help='The maximum number of weights saved in the cache_dict. You can adjust this'
                         ' number based on your GPU memory. Default is 200.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    splitted_weights_path = args.splitted_weights_path

    if args.split_weight:
        os.makedirs(splitted_weights_path, exist_ok=True)
        LowMemoryLlama.split_and_convert_to_cpu_int4_weights(model_path, splitted_weights_path)

    model = LowMemoryLlama(model_path, splitted_weights_path, args.max_cache_num)

    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = model.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
