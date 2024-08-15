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
# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
#

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
import os
import time
import numpy as np
from typing import Callable, List, Optional, Union, Tuple
from types import SimpleNamespace
import transformers
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ipex_llm.utils.common import invalidInputError
from ipex_llm.ggml.quantize import ggml_tensor_qtype
import logging
logger = logging.getLogger(__name__)

# patch GenerationMixin.generate
from transformers import GenerationMixin
original_generate = GenerationMixin.generate


class DummyLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/
        # python/llm/src/ipex_llm/transformers/models/llama.py#L2076
        self.weight = nn.Parameter(torch.empty(0,), requires_grad=False)

    def forward(self, x):
        return x


class Dummy_MLPLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/
        # python/llm/src/ipex_llm/transformers/models/llama.py#L119
        self.up_proj = DummyLayer()
        self.down_proj = DummyLayer()
        self.shared_expert = SimpleNamespace()
        self.shared_expert.up_proj = DummyLayer()

    def forward(self, x):
        return x


class Dummy_DecoderLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError
        self.input_layernorm = DummyLayer()
        self.mlp = Dummy_MLPLayer()

    def forward(self, hidden_states, *args, **kwargs):
        past_key_value = kwargs.get('past_key_value', None)
        use_cache = kwargs.get('use_cache', False)
        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class Dummy_GLMBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError
        self.input_layernorm = DummyLayer()
        self.mlp = Dummy_MLPLayer()

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        if kv_cache is None:
            return hidden_states, ()
        return hidden_states, kv_cache


def init_pipeline_parallel():
    import oneccl_bindings_for_pytorch
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group('ccl')


def low_mem_convert(model):
    from ipex_llm.transformers.convert import convert_forward
    import importlib
    if 'llama' in model.config.model_type:
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaForCausalLM,
            llama_causallm_forward_4_37_lowmem)
    elif model.config.model_type == "chatglm" and not hasattr(model.config, "vision_config"):
        if model.config.num_layers == 40:
            # for glm4-9b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(
                model,
                module.ChatGLMForConditionalGeneration,
                glm4_conditional_generation_forward_lowmem)
        else:
            # for chatglm3-6b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(
                model,
                module.ChatGLMForConditionalGeneration,
                chatglm3_conditional_generation_forward_lowmem)
    return model


def pipeline_parallel(model, pipeline_parallel_stages, torch_dtype=torch.float32, device=None):
    global num_layers
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'num_layers'):
        # for chatglm3-6b
        num_layers = model.config.num_layers

    slice_size = (num_layers + pipeline_parallel_stages - 1) // pipeline_parallel_stages

    local_rank = dist.get_rank()

    global layer_start
    global layer_end
    layer_start = slice_size * local_rank
    layer_end = layer_start + min(slice_size, num_layers - layer_start)

    if model.config.model_type == "qwen" and hasattr(model.config, "visual"):
        # for Qwen-VL-Chat
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['transformer'].h[i] = Dummy_DecoderLayer()
        if local_rank != 0:
            model._modules['transformer'].wte = DummyLayer()
            model._modules['transformer'].drop = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['transformer'].ln_f = DummyLayer()
            model._modules['ln_f'] = DummyLayer()
            model._modules['lm_head'] = DummyLayer()
    elif model.config.model_type == "chatglm":
        # for chatglm3-6b, glm-4-9b-chat
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['transformer'].encoder.layers[i] = Dummy_GLMBlock()
            else:
                model._modules['transformer'].encoder.layers[i].self_attention.num_layers = \
                    i - layer_start

        if local_rank != 0:
            model._modules['transformer'].embedding = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['transformer'].encoder.final_layernorm = DummyLayer()
            model._modules['transformer'].output_layer = DummyLayer()
    else:
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['model'].layers[i] = Dummy_DecoderLayer()
            else:
                model._modules['model'].layers[i].self_attn.layer_idx = i - layer_start

        if local_rank != 0:
            model._modules['model'].embed_tokens = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['model'].norm = DummyLayer()
            model._modules['lm_head'] = DummyLayer()

    _enable_lowmem = os.getenv('IPEX_LLM_LOW_MEM')
    _enable_lowmem = (_enable_lowmem is not None) and (_enable_lowmem.lower() == "1")
    if _enable_lowmem:
        model = low_mem_convert(model)

    model.pipeline_parallel_stages = pipeline_parallel_stages
    model.layer_start = layer_start
    model.layer_end = layer_end
    model.num_layers = num_layers
    if torch_dtype == torch.float16:
        model = model.half()
    if device is None:
        model = model.to(f'xpu:{local_rank}')
    else:
        model.to(device)
    return model


@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
    if hasattr(self, 'pipeline_parallel_stages') and self.pipeline_parallel_stages > 1:
        # priority: `generation_config` argument > `model.generation_config`
        if generation_config is None:
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning("Setting `pad_token_id` to `eos_token_id`: "
                           f"{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config is not None and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.pop("max_new_tokens")
        else:
            max_new_tokens = kwargs.pop("max_new_tokens", None)

        return self.pipeline_parallel_generate(inputs=inputs,
                                               max_new_tokens=max_new_tokens,
                                               generation_config=generation_config,
                                               **kwargs)

    return original_generate(self,
                             inputs=inputs,
                             generation_config=generation_config,
                             logits_processor=logits_processor,
                             stopping_criteria=stopping_criteria,
                             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                             synced_gpus=synced_gpus,
                             assistant_model=assistant_model,
                             streamer=streamer,
                             **kwargs)

GenerationMixin.generate = generate


@torch.no_grad()
def pipeline_parallel_generate(self,
                               inputs: Optional[torch.Tensor] = None,
                               max_new_tokens: int = 32,
                               generation_config: Optional[GenerationConfig] = None,
                               **kwargs):
    model_kwargs = generation_config.update(**kwargs)
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    bs = inputs_tensor.shape[0]
    if model_kwargs.get("attention_mask", None) is None:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=bs,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" \
            else model_kwargs.pop("input_ids")

    local_rank = dist.get_rank()
    pre_rank = (local_rank - 1) % self.pipeline_parallel_stages
    next_rank = (local_rank + 1) % self.pipeline_parallel_stages

    global layer_start
    global layer_end
    global num_layers

    self.first_token_time = 0
    self.next_token_time = []

    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) \
        if eos_token_id is not None else None

    _input_ids = None
    _past_key_values = None

    bs = input_ids.shape[0]
    output_ids = input_ids.clone()
    os.environ["IPEX_LLM_QUANTIZE_KV_CACHE"] = "0"

    step = 0
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    while True:
        if step >= max_new_tokens:
            break

        if _input_ids is None:
            _input_ids = input_ids

        model_inputs = self.prepare_inputs_for_generation(output_ids, **model_kwargs)

        tic = time.time()
        if local_rank == 0:
            outputs = self(**model_inputs)
        else:
            _inputs_shape = _input_ids.shape + (self.config.hidden_size,)
            if step == 0 and self.config.model_type == "chatglm" \
               and hasattr(self.config, "vision_config"):
                # for glm-4v, image features are mapped during 1st token
                # 1597 are computed according to computation process of conv
                _images_feature = 1597 + _input_ids.shape[0] * 2 + _input_ids.shape[1]
                _inputs_shape = (_input_ids.shape[0], _images_feature, self.config.hidden_size,)
            inputs_embeds = torch.empty(_inputs_shape,
                                        device=input_ids.device, dtype=torch.float16)
            dist.recv(inputs_embeds, src=pre_rank)
            model_inputs.pop("input_ids")
            model_inputs["inputs_embeds"] = inputs_embeds
            outputs = self(**model_inputs)

        if local_rank == self.pipeline_parallel_stages - 1:
            logits = outputs.logits
            next_ids = torch.argmax(logits[:, -1:, :], dim=-1)
            dist.broadcast(next_ids, src=local_rank)
        else:
            send_data = outputs[0].to(torch.float16)
            dist.send(send_data, dst=next_rank)
            next_ids = torch.empty((bs, 1), device=input_ids.device, dtype=torch.int64)
            dist.broadcast(next_ids, src=self.pipeline_parallel_stages - 1)

        _input_ids = next_ids
        output_ids = torch.cat([output_ids, next_ids], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # finished sentences should have their next token be a padding token
        next_ids = next_ids.squeeze()
        if eos_token_id is not None:
            if pad_token_id is None:
                invalidInputError(False, "If `eos_token_id` is defined, "
                                         "make sure that `pad_token_id` is defined.")
            next_ids = next_ids * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if self.config.model_type == "chatglm" and self.config.num_layers == 40 \
           and not hasattr(self.config, "vision_config"):
            # for glm-4-9b-chat
            if step == 0:
                value_placeholder = torch.empty_like((outputs.past_key_values)[-1][0])
                past_key_values_placeholder = tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_start)
                ) + (outputs.past_key_values)[: layer_end - layer_start] + tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_end, num_layers)
                )
                _past_key_values = past_key_values_placeholder
            else:
                _past_key_values = outputs.past_key_values
        elif self.config.model_type in ["baichuan", "chatglm"] or \
                (self.config.model_type == "qwen" and hasattr(self.config, "visual")):
            # for baichuan2, chatglm3, Qwen-VL-Chat, glm-4v-9b
            if local_rank != 0:
                value_placeholder = torch.empty_like((outputs.past_key_values)[-1][0])
                past_key_values_placeholder = tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_start)
                ) + (outputs.past_key_values)[layer_start:]
                _past_key_values = past_key_values_placeholder
            else:
                _past_key_values = outputs.past_key_values
        else:
            _past_key_values = outputs.past_key_values

        toc = time.time()
        if step == 0:
            self.first_token_time = toc - tic
        else:
            self.next_token_time.append(toc - tic)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_ids.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        if this_peer_finished:
            break

        step += 1
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
    self.rest_cost_mean = np.mean(self.next_token_time)
    return output_ids


def llama_causallm_forward_4_37_lowmem(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    # ipex-llm change starts

    device = hidden_states.device

    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)  # noqa
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]  # noqa
        logits = torch.cat(logits, dim=-1)
    else:
        if device.type == "xpu":
            torch.xpu.empty_cache()
        logits = self.lm_head(hidden_states)
        if device.type == "xpu":
            torch.xpu.empty_cache()
    # logits = logits.float()

    # ipex-llm change ends

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def chatglm3_conditional_generation_forward_lowmem(
    self,
    input_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[-1:]

    device = hidden_states.device
    # ipex-llm change starts
    if device.type == "xpu":
        torch.xpu.empty_cache()
    lm_logits = self.transformer.output_layer(hidden_states)
    if device.type == "xpu":
        torch.xpu.empty_cache()
    lm_logits = lm_logits.transpose(0, 1).contiguous()

    loss = None
    if labels is not None:
        # lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
    # ipex-llm change ends

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def glm4_conditional_generation_forward_lowmem(
    self,
    input_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[:, -1:]

    device = hidden_states.device
    # ipex-llm change starts
    if device.type == "xpu":
        torch.xpu.empty_cache()
    lm_logits = self.transformer.output_layer(hidden_states)
    if device.type == "xpu":
        torch.xpu.empty_cache()

    loss = None
    if labels is not None:
        # lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
    # ipex-llm change ends

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
