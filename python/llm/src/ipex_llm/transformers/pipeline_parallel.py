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
import torch.distributed as dist
import os
import time
import numpy as np
from typing import Callable, List, Optional
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ipex_llm.utils.common import invalidInputError
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
        self.weight = torch.randn(1,)

    def forward(self, x):
        return x


class Dummy_MLPLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/
        # python/llm/src/ipex_llm/transformers/models/llama.py#L119
        self.up_proj = DummyLayer()
        self.down_proj = DummyLayer()

    def forward(self, x):
        return x


class Dummy_DecoderLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError
        self.input_layernorm = DummyLayer()
        self.mlp = Dummy_MLPLayer()

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **kwargs):
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
        return hidden_states, kv_cache


def init_pipeline_parallel():
    import oneccl_bindings_for_pytorch
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group('ccl')


def pipeline_parallel(model, pipeline_parallel_stages):
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

    if model.config.architectures is not None \
       and model.config.architectures[0] in ["ChatGLMModel", "ChatGLMForConditionalGeneration"]:
        # for chatglm3-6b
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

    model.pipeline_parallel_stages = pipeline_parallel_stages
    model = model.to(f'xpu:{local_rank}')
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
            max_new_tokens = generation_config.max_new_tokens
        else:
            max_new_tokens = kwargs.get("max_new_tokens", None)

        return self.pipeline_parallel_generate(inputs=inputs,
                                               max_new_tokens=max_new_tokens,
                                               generation_config=generation_config,)

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
    eos_token_id_tensor = torch.tensor(eos_token_id).to(inputs.device) \
        if eos_token_id is not None else None

    _input_ids = None
    _past_key_values = None
    bs = inputs.shape[0]
    output_ids = inputs.clone()

    step = 0
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(inputs.shape[0], dtype=torch.long, device=inputs.device)
    this_peer_finished = False
    while True:
        if step >= max_new_tokens:
            break

        if _input_ids is None:
            _input_ids = inputs

        tic = time.time()
        if local_rank == 0:
            outputs = self(input_ids=_input_ids, inputs_embeds=None,
                           past_key_values=_past_key_values, use_cache=True)
        else:
            inputs_embeds = torch.empty(_input_ids.shape + (self.config.hidden_size,),
                                        device=f'xpu:{local_rank}', dtype=self.dtype)
            dist.recv(inputs_embeds, src=pre_rank)
            outputs = self(input_ids=None, inputs_embeds=inputs_embeds,
                           past_key_values=_past_key_values, use_cache=True)

        if local_rank == self.pipeline_parallel_stages - 1:
            logits = outputs.logits
            next_ids = torch.argmax(logits[:, -1:, :], dim=-1)
            dist.broadcast(next_ids, src=local_rank)
        else:
            dist.send(outputs[0].to(self.dtype), dst=next_rank)
            next_ids = torch.empty((bs, 1), device=f'xpu:{local_rank}', dtype=torch.int64)
            dist.broadcast(next_ids, src=self.pipeline_parallel_stages - 1)

        _input_ids = next_ids
        output_ids = torch.cat([output_ids, next_ids], dim=-1)

        # finished sentences should have their next token be a padding token
        next_ids = next_ids.squeeze()
        if eos_token_id is not None:
            if pad_token_id is None:
                invalidInputError(False, "If `eos_token_id` is defined, "
                                         "make sure that `pad_token_id` is defined.")
            next_ids = next_ids * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # Temporarily specify as Baichuan and ChatGLM
        if self.config.model_type in ["baichuan", "chatglm"] and local_rank != 0:
            value_placeholder = torch.empty_like((outputs.past_key_values)[-1][0])
            past_key_values_placeholder = tuple(
                (value_placeholder, value_placeholder) for _ in range(layer_start)
            ) + (outputs.past_key_values)[layer_start:]
            _past_key_values = past_key_values_placeholder
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
