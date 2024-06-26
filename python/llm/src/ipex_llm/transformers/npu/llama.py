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


import torch


import torch.utils
import torch
from torch import nn
import numpy as np
import transformers
from ipex_llm.transformers.npu.utils import get_npu_model
from typing import Optional, Tuple
import torch.nn.functional as F
from typing import List, Union
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaConfig, LlamaRMSNorm
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, logger
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
from ipex_llm.transformers.convert import convert_forward
from ipex_llm.utils.common.log4Error import invalidInputError
import transformers
import time


class NPUModelDecode(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states, attention_mask, position_ids, past_key_values, weights):

        # msg = "sanity check"
        # for i in range(len(self.layers)):
        #     layer_weights = weights[i]
        #     invalidInputError(layer_weights[0] is self.layers[i].self_attn.q_proj.weight, msg)
        #     invalidInputError(layer_weights[1] is self.layers[i].self_attn.k_proj.weight, msg)
        #     invalidInputError(layer_weights[2] is self.layers[i].self_attn.v_proj.weight, msg)
        #     invalidInputError(layer_weights[3] is self.layers[i].self_attn.o_proj.weight, msg)
        #     invalidInputError(layer_weights[4] is self.layers[i].mlp.gate_proj.weight, msg)
        #     invalidInputError(layer_weights[5] is self.layers[i].mlp.up_proj.weight, msg)
        #     invalidInputError(layer_weights[6] is self.layers[i].mlp.down_proj.weight, msg)

        next_cache = []
        for i, layer in enumerate(self.layers):
            outputs = layer(hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values[i],
                            output_attentions=False,
                            use_cache=True)
            hidden_states = outputs[0]
            next_cache.append(outputs[1])
        return hidden_states, next_cache


class NPUModelPrefill(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states, attention_mask, position_ids, weights):

        # msg = "sanity check"
        # for i in range(len(self.layers)):
        #     layer_weights = weights[i]
        #     invalidInputError(layer_weights[0] is self.layers[i].self_attn.q_proj.weight, msg)
        #     invalidInputError(layer_weights[1] is self.layers[i].self_attn.k_proj.weight, msg)
        #     invalidInputError(layer_weights[2] is self.layers[i].self_attn.v_proj.weight, msg)
        #     invalidInputError(layer_weights[3] is self.layers[i].self_attn.o_proj.weight, msg)
        #     invalidInputError(layer_weights[4] is self.layers[i].mlp.gate_proj.weight, msg)
        #     invalidInputError(layer_weights[5] is self.layers[i].mlp.up_proj.weight, msg)
        #     invalidInputError(layer_weights[6] is self.layers[i].mlp.down_proj.weight, msg)

        next_cache = []
        for i, layer in enumerate(self.layers):
            outputs = layer(hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=True)
            hidden_states = outputs[0]
            next_cache.append(outputs[1])
        return hidden_states, next_cache

def flatten(inputs):
    if isinstance(inputs, (list, tuple)):
        result = []
        for i in inputs:
            result.extend(flatten(i))
        return result
    else:
        return [inputs]


def gather_weights_from_decoder(decoder, qtype):
    attn = decoder.self_attn
    mlp = decoder.mlp
    if qtype in supporte_qtype:

        return [
            (attn.q_proj.weight, attn.q_proj.scale),
            (attn.k_proj.weight, attn.k_proj.scale),
            (attn.v_proj.weight, attn.v_proj.scale),
            (attn.o_proj.weight, attn.o_proj.scale),
            (mlp.gate_proj.weight, mlp.gate_proj.scale),
            (mlp.up_proj.weight, mlp.up_proj.scale),
            (mlp.down_proj.weight, mlp.down_proj.scale)
        ]
    else: 
        return [
            attn.q_proj.weight,
            attn.k_proj.weight,
            attn.v_proj.weight,
            attn.o_proj.weight,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight]

from ipex_llm.transformers.npu.quantization import compress_to_i4
import openvino as ov
def set_weights_for_decoder(request, npu_decoder,
                            weights, num_layers,
                            weights_arg_offset,
                            qtype=None):
    if qtype in supporte_qtype:
        for i in range(num_layers):
            layer_weights = weights[i]
            for j in range(7):
                w, scale = layer_weights[j]
                offset = weights_arg_offset + i * 7 * 2 + j * 2
                w_port = npu_decoder.input(offset)
                w_npu_tensor = request.get_tensor(w_port)
                if qtype == "sym_int4":
                    w_int4 = compress_to_i4(w)
                    w = ov.Tensor(w_int4.numpy(), w.shape, ov.Type.i4)
                w_npu_tensor.copy_from(w)

                scale_port = npu_decoder.input(offset + 1)
                scale_npu_tensor = request.get_tensor(scale_port)
                scale_npu_tensor.copy_from(scale.numpy())
    else:
        for i in range(num_layers):
            layer_weights = weights[i]
            for j in range(7):
                w = layer_weights[j]
                w_port = npu_decoder.input(weights_arg_offset + i * 7 + j)
                w_npu_tensor = request.get_tensor(w_port)
                w_npu_tensor.copy_from(w.numpy())


supporte_qtype = {"sym_int8", "sym_int4"}

def offload_llama_decoder_to_npu(model,
                                 max_prompt_size,
                                 max_output_size,
                                 qtype=None,
                                 num_layers=None):
    
    if qtype in supporte_qtype:
        from ipex_llm.transformers.npu.quantization import lower_linear
        start = time.perf_counter()
        lower_linear(model.model.layers, qtype)
        end = time.perf_counter()
        print(f"done quantization, using time {end - start}s")

    llama_model = model.model
    if num_layers is None:
        num_layers = llama_model.config.num_hidden_layers

    kv_cache_len_max = max_prompt_size + max_output_size

    hidden_size = llama_model.config.hidden_size
    num_heads = llama_model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = llama_model.config.num_key_value_heads
    def get_decoder_model(layers, hidden_size, num_heads, head_dim, num_key_value_heads):
        with torch.no_grad():
            npu_torch_model = NPUModelDecode(layers)
            layers_weights = [gather_weights_from_decoder(layers[i], qtype=qtype)
                            for i in range(num_layers)]
            kv_cache = [(torch.randn(1, num_key_value_heads, kv_cache_len_max, head_dim,
                                    dtype=torch.float16),
                        torch.randn(1, num_key_value_heads, kv_cache_len_max, head_dim,
                                    dtype=torch.float16)
                        ) for i in range(num_layers)]
            dummy_inputs = (torch.randn(1, 1, hidden_size, dtype=torch.float16),
                            torch.ones(1, 1, 1, kv_cache_len_max + 1, dtype=torch.float16),
                            torch.ones(1, 1, dtype=torch.long) * kv_cache_len_max,
                            kv_cache,
                            layers_weights,
                            )
            input_names = ["input", "attention_mask", "position_ids", "past_key_values", "weights"]
            output_names = ["output", "new_past_key_values"]
            
            # get int4 weights indexes in the inputs
            weights_arg_offset = 3 + len(kv_cache) * 2
            indexes = None
            if qtype == "sym_int4":
                indexes = [weights_arg_offset + i * 7 * 2 for i in range(num_layers)]
            
            npu_decoder, core = get_npu_model(npu_torch_model, dummy_inputs,
                                            input_names, output_names, quantize=False,
                                            device="NPU", int4_indexes=indexes)
            infer_request = npu_decoder.create_infer_request()
            
            set_weights_for_decoder(infer_request, npu_decoder, layers_weights, num_layers,
                                    weights_arg_offset=weights_arg_offset, qtype=qtype)
        return infer_request, npu_decoder
    
    def get_prefill_model(layers, hidden_size, num_heads, head_dim, num_key_value_heads):
        with torch.no_grad():
            npu_torch_model = NPUModelPrefill(layers)
            layers_weights = [gather_weights_from_decoder(layers[i], qtype=qtype)
                            for i in range(num_layers)]
            dummy_inputs = (torch.ones(1, max_prompt_size, hidden_size, dtype=torch.float16) / 0,
                            torch.ones(1, 1, max_prompt_size, max_prompt_size, dtype=torch.float16),
                            torch.ones(1, max_prompt_size, dtype=torch.long) * (max_prompt_size - 1),
                            layers_weights,
                            )
            input_names = ["input", "attention_mask", "position_ids", "weights"]
            output_names = ["output", "new_past_key_values"]
            
            weights_arg_offset = 3
            indexes = None
            if qtype == "sym_int4":
                indexes = [weights_arg_offset + i * 7 * 2 for i in range(num_layers)]
            npu_decoder, core = get_npu_model(npu_torch_model, dummy_inputs,
                                            input_names, output_names, quantize=False,
                                            device="NPU", int4_indexes=indexes)
            infer_request = npu_decoder.create_infer_request()
            
            set_weights_for_decoder(infer_request, npu_decoder, layers_weights, num_layers,
                                    weights_arg_offset=weights_arg_offset, 
                                    qtype=qtype)
        return infer_request, npu_decoder
    
    layers = llama_model.layers[:num_layers].to(torch.float16)
    start = time.perf_counter()
    infer_request_prefill, npu_prefill_model = get_prefill_model(layers,
                                                               hidden_size, num_heads, head_dim, num_key_value_heads)
    end = time.perf_counter()
    print(f"get prefill model, using time {end - start}s")
    
    start = time.perf_counter()
    infer_request_decode, npu_decode_model = get_decoder_model(layers,
                                                               hidden_size, num_heads, head_dim, num_key_value_heads)
    end = time.perf_counter()
    print(f"get decoder model, using time {end - start}s")


    def padding_decoder_inputs(hidden_states, attention_mask, position_ids,
                               past_key_values, pad_len):
        pad_mask = (pad_len, 0)
        pad_kv_cache = (0, 0, pad_len, 0)
        padded_attention_mask = F.pad(attention_mask.to(torch.float16), pad_mask,
                                      value=torch.finfo(torch.float16).min)
        real_inputs = (hidden_states.to(torch.float16),
                       padded_attention_mask,
                       position_ids,
                       [(F.pad(past_key_values[i][0].to(torch.float16), pad_kv_cache, value=0.0),
                         F.pad(past_key_values[i][1].to(torch.float16), pad_kv_cache, value=0.0))
                        for i in range(num_layers)
                        ])
        return flatten(real_inputs)
    
    def padding_prefill_inputs(hidden_states, attention_mask, position_ids,
                               past_key_values, pad_len):
        pad_mask = (pad_len, 0, pad_len, 0)
        padded_attention_mask = F.pad(attention_mask.to(torch.float16), pad_mask,
                                      value=torch.finfo(torch.float16).min)
        pad_hidden_states = (0, 0, pad_len, 0)
        padded_hidden_states = F.pad(hidden_states.to(torch.float16), pad_hidden_states,
                                     value=0.0)
        pad_p_ids = (pad_len, 0)
        padded_position_ids = F.pad(position_ids, pad_p_ids, value=0)
        real_inputs = (padded_hidden_states,
                       padded_attention_mask,
                       padded_position_ids)
        return flatten(real_inputs)
    
    def npu_inference(npu_model, ov_infer_req, padding_fn, get_pad_len_fn,
                      hidden_states, attention_mask, position_ids, past_key_values):
        next_decoder_cache = ()
        pad_len = get_pad_len_fn(attention_mask)
        padded_inputs = padding_fn(hidden_states=hidden_states,
                                               attention_mask=attention_mask,
                                               position_ids=position_ids,
                                               past_key_values=past_key_values,
                                               pad_len=pad_len)

        for i in range(len(padded_inputs)):
            input_port = npu_model.input(i)
            import openvino as ov
            ov_infer_req.set_tensor(input_port, ov.Tensor(padded_inputs[i].numpy(),
                                                           shared_memory=True))
        npu_output = ov_infer_req.infer()
        hidden_states = torch.from_numpy(np.array(npu_output[0].data, copy=False))
        hidden_states = hidden_states.to(hidden_states.dtype)

        for i in range(0, num_layers):
            k = torch.from_numpy(np.array(npu_output[2*i + 1].data, copy=False))[:, :, pad_len:, :]
            v = torch.from_numpy(np.array(npu_output[2*i + 2].data, copy=False))[:, :, pad_len:, :]
            next_decoder_cache += ((k.to(hidden_states.dtype), v.to(hidden_states.dtype)),)
        return hidden_states, next_decoder_cache

    def npu_decode_inference(hidden_states, attention_mask, position_ids, past_key_values):
        return npu_inference(npu_model=npu_decode_model,
                             ov_infer_req=infer_request_decode,
                      padding_fn=padding_decoder_inputs,
                      get_pad_len_fn=lambda x: kv_cache_len_max + 1 - attention_mask.size(-1),
                      hidden_states=hidden_states,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values
                      )
    
    def npu_prefill_inference(hidden_states, attention_mask, position_ids, past_key_values):
        return npu_inference(npu_model=npu_prefill_model,
                             ov_infer_req=infer_request_prefill,
                      padding_fn=padding_prefill_inputs,
                      get_pad_len_fn=lambda x: max_prompt_size - attention_mask.size(-1),
                      hidden_states=hidden_states,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values
                      )

    def llama_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            invalidInputError(False,
                              ("You cannot specify both input_ids "
                               "and inputs_embeds at the same time"))
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            invalidInputError(False, "You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (
                attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds,
                past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. "
                    "Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        ###########################################################################################
        # Begin NPU additional logic
        query_length = hidden_states.size(1)
        if num_layers > 0:
            invalidInputError(not output_attentions,
                              "output_attentions is not supported when offloading to NPU")
            invalidInputError(not output_hidden_states,
                              "output_hidden_states is not supported when offloading to NPU")
            bs = hidden_states.size(0)
            invalidInputError(bs == 1,
                              "batch size must be 1 when offloading to NPU")
            invalidInputError(use_cache, "use_cache is not supported when offloading to NPU")
            if query_length == 1:
                hidden_states, next_decoder_cache = npu_decode_inference(hidden_states, attention_mask,
                                                                  position_ids, past_key_values)
            else:
                hidden_states, next_decoder_cache = npu_prefill_inference(hidden_states, attention_mask,
                                                                  position_ids, past_key_values)
        # End NPU additional logic
        ###########################################################################################

        for idx, decoder_layer in enumerate(self.layers):
            #######################################################################################
            # Begin NPU additional logic
            if num_layers > 0 and idx < num_layers:
                continue
            # End NPU additional logic
            #######################################################################################
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache,
                                     all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    convert_forward(model,
                    transformers.models.llama.modeling_llama.LlamaModel,
                    llama_model_forward)
    return model

