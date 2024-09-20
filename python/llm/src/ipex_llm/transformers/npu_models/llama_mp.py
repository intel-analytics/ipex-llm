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
import torch
import time

from typing import Optional, Sequence, List, Union, Any, Tuple
import numpy as np

from transformers.cache_utils import Cache
from ipex_llm.utils.common import invalidInputError
from typing import Optional, List, Generator
import uuid
from functools import partial
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist

from transformers.utils import logging

logger = logging.get_logger(__name__)
from colorama import Fore, Back, Style
import torch.multiprocessing as mp
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from ipex_llm.transformers.npu_models.mp_models_base import BasePrefillRunner
from ipex_llm.transformers.npu_models.mp_models_base import BaseDecodeRunner
from ipex_llm.transformers.npu_models.mp_models_base import llama
from ipex_llm.transformers.npu_models.common import reshape_lm_head_input
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss


class DecodeRunner(BaseDecodeRunner):
    def __init__(self, model, max_seq_len, intra_pp=2, inter_pp=2, transpose_value_cache=True):
        super().__init__(model, max_seq_len, intra_pp, inter_pp, transpose_value_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):

        if self.cache_past_key_value != past_key_value:
            control = torch.tensor(-1, dtype=torch.int)
            dist.broadcast(control, src=0)
            for i in range(len(self.decoder_processes)):
                self.input_queues[i].put(past_key_value)
        dist.broadcast(self.forward_signal, src=0, async_op=True)
        hidden_states = hidden_states.to(torch.float16)
        dist.send(hidden_states, dst=1)
        past_key_value.expand(self.transpose_value_cache)
        dist.recv(hidden_states, src=self.world_size - 1)
        return hidden_states, past_key_value


class PrefillRunner(BasePrefillRunner):
    def __init__(self, model, max_output_len, max_prompt_len, transpose_value_cache):
        super().__init__(model, max_output_len, max_prompt_len, transpose_value_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        seq_len = hidden_states.size(1)
        invalidInputError(
            seq_len <= self.max_prompt_len,
            (
                f"seq_len: {seq_len} should be less than or equal"
                " to max_prompt_len {self.max_prompt_len}"
            ),
        )
        pad_len = self.max_prompt_len - seq_len
        hidden_states = F.pad(hidden_states.to(torch.float16), (0, 0, 0, pad_len), value=0.0)
        position_ids = F.pad(position_ids, (0, pad_len), value=0)
        attention_mask = F.pad(
            attention_mask.to(torch.float16),
            (0, pad_len, 0, pad_len),
            value=torch.finfo(torch.float16).min,
        )

        args = (hidden_states, position_ids, attention_mask, past_key_value, cache_position)
        self.prefill_input_queue.put(args)
        hidden_states, past_key_value = self.prefill_result_queue.get()
        past_key_value.shrink(seq_len, self.transpose_value_cache)
        hidden_states = hidden_states[:, :seq_len, :]
        return hidden_states, past_key_value


def gen_llama_fused_model_forward(prefill_runner, decode_runner):

    def llama_fused_model_forward(
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
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            msg = (
                "You cannot specify both input_ids and inputs_embeds at the same time,"
                " and must specify either one"
            )
            invalidInputError(False, msg)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0

        # ipex-llm changes start
        from ipex_llm.transformers.npu_models.kv import DynamicFusedNormalCache

        if use_cache and not isinstance(past_key_values, DynamicFusedNormalCache):
            past_key_values = DynamicFusedNormalCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        # ipex-llm changes end

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        seq_len = hidden_states.size(1)

        if seq_len == 1:
            layers_runner = decode_runner
        else:
            layers_runner = prefill_runner
        layer_outputs = layers_runner.forward(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]

        next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ipex-llm changes start
        next_cache = next_decoder_cache if use_cache else None
        # ipex-llm changes end
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return llama_fused_model_forward


def llama2_casullm_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
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
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # ipex-llm change start
    hidden_states = reshape_lm_head_input(hidden_states)
    # ipex-llm change end
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp,
                                                   dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i])
                  for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

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
