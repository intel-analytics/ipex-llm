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


def init_pipeline_parallel():
    import oneccl_bindings_for_pytorch
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group('ccl')


def pipeline_parallel(model, pipeline_parallel_stages):
    slice_size = (model.config.num_hidden_layers + pipeline_parallel_stages - 1) // \
        pipeline_parallel_stages

    local_rank = dist.get_rank()
    layer_start = slice_size * local_rank
    layer_end = layer_start + min(slice_size, model.config.num_hidden_layers - layer_start)

    for i in range(model.config.num_hidden_layers):
        if i < layer_start or i >= layer_end:
            model._modules['model'].layers[i] = Dummy_DecoderLayer()
        else:
            # align layer_idx and len(past_key_values), otherwise abnormal output
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
        if generation_config is not None and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.max_new_tokens
        else:
            max_new_tokens = kwargs.get("max_new_tokens", None)
        return self.pipeline_parallel_generate(inputs=inputs,
                                               max_new_tokens=max_new_tokens,)

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
                               **kwargs):
    local_rank = dist.get_rank()
    pre_rank = (local_rank - 1) % self.pipeline_parallel_stages
    next_rank = (local_rank + 1) % self.pipeline_parallel_stages

    self.first_token_time = 0
    self.next_token_time = []

    _input_ids = None
    _past_key_values = None
    bs = inputs.shape[0]
    output_ids = inputs.clone()

    step = 0
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
                                        device=f'xpu:{local_rank}', dtype=torch.float32)
            dist.recv(inputs_embeds, src=pre_rank)
            outputs = self(input_ids=None, inputs_embeds=inputs_embeds,
                           past_key_values=_past_key_values, use_cache=True)

        if local_rank == self.pipeline_parallel_stages - 1:
            logits = outputs.logits
            next_ids = torch.argmax(logits[:, -1:, :], dim=-1)
            dist.broadcast(next_ids, src=local_rank)
        else:
            dist.send(outputs[0], dst=next_rank)
            next_ids = torch.empty((bs, 1), device=f'xpu:{local_rank}', dtype=torch.int64)
            dist.broadcast(next_ids, src=self.pipeline_parallel_stages - 1)

        _input_ids = next_ids
        output_ids = torch.cat([output_ids, next_ids], dim=-1)
        _past_key_values = outputs.past_key_values
        toc = time.time()
        if step == 0:
            self.first_token_time = toc - tic
        else:
            self.next_token_time.append(toc - tic)
        step += 1
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
    self.rest_cost_mean = np.mean(self.next_token_time)
    return output_ids
