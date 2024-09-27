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
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation
# /utils.py
#

from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import torch
import time
import numpy as np
import random
import logging
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ipex_llm.transformers.speculative import greedy, deepmind_sample, logits_to_probs,\
    _crop_past_key_values, _prepare_generate_args
from ipex_llm.utils.common import invalidInputError
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger("ipex_llm.npu")

# patch GenerationMixin.generate
from transformers import GenerationMixin
original_generate = GenerationMixin.generate
query_group_size = 16


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
    if kwargs.get("num_beams", None) not in [None, 1]:
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
    else:
        return self.npu_generate(inputs=inputs,
                                 generation_config=generation_config,
                                 streamer=streamer,
                                 logits_processor=logits_processor,
                                 stopping_criteria=stopping_criteria,
                                 prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                 **kwargs)


GenerationMixin.generate = generate


def clear_benchmarks(self):
    self.first_token_time = None
    self.last_token_time = []
    self.encoder_time = 0


@torch.no_grad()
def npu_generate(self,
                 inputs: Optional[torch.Tensor] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 streamer: Optional["BaseStreamer"] = None,
                 **sampling_kwargs):
    input_ids, generation_config, logits_processor, stopping_criteria, \
        model_kwargs = _prepare_generate_args(self, inputs, generation_config,
                                              logits_processor=logits_processor,
                                              stopping_criteria=stopping_criteria,
                                              streamer=streamer,
                                              generate_attention_mask=True,
                                              **sampling_kwargs)

    step = 0
    max_new_tokens = generation_config.max_new_tokens
    attn_mask = model_kwargs.get("attention_mask", None)

    if self.config.model_type == "qwen2":
        from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
        docoder_logits_processor = LogitsProcessorList([item for item in logits_processor \
                                                        if not isinstance(item, MinNewTokensLengthLogitsProcessor)])
    else:
        docoder_logits_processor = LogitsProcessorList()

    clear_benchmarks(self)

    input_len = input_ids.shape[1]

    eos_token_id_set = None
    if generation_config.eos_token_id is not None:
        if isinstance(generation_config.eos_token_id, list):
            eos_token_id_set = set(generation_config.eos_token_id)
        else:
            eos_token_id_set = set([generation_config.eos_token_id])

    while True:
        if step >= max_new_tokens:
            break

        tic = time.perf_counter()

        if step == 0:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            output = self(**model_inputs,
                          return_dict=True)
            # output = CausalLMOutputWithPast(
            #     loss=None,
            #     logits=torch.empty([1, 1, self.config.vocab_size]),
            #     past_key_values=None,
            #     hidden_states=None,
            #     attentions=None,
            # )
            logits = output['logits'][:, -1:]
            logits[:, -1, :] = logits_processor(input_ids, logits[:, -1, :])
            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
            else:
                output_ids = torch.argmax(logits, dim=-1)
            input_ids = torch.cat((input_ids, output_ids), dim=-1)
        else:
            model_inputs = {
                "input_ids": input_ids[:, -1:],
                "past_key_values": past_key_values,
                # "position_ids": model_kwargs["position_ids"],
                "use_cache": True,
                "attention_mask": attn_mask,
            }

            output = output = self(**model_inputs,
                                   return_dict=True)
            # output = CausalLMOutputWithPast(
            #     loss=None,
            #     logits=torch.empty([1, 1, self.config.vocab_size]),
            #     past_key_values=None,
            #     hidden_states=None,
            #     attentions=None,
            # )
            t1 = time.perf_counter()

            logits = output['logits']

            logits[:, -1, :] = docoder_logits_processor(input_ids,
                                                        logits[:, -1, :])
            t2 = time.perf_counter()

            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
                output_ids = output_ids.transpose(0, 1)
            else:
                output_ids = torch.argmax(logits, dim=-1)
            
            t3 = time.perf_counter()

            input_ids = torch.cat((input_ids, output_ids), dim=-1)
            t4 = time.perf_counter()

        step += 1

        past_key_values = output.past_key_values
        if attn_mask is not None:
            attn_mask = torch.cat(
            [attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1
        )

        toc = time.perf_counter()

        if self.first_token_time is None:
            self.first_token_time = toc - tic
        else:
            self.last_token_time.append(toc - tic)
            print(f"Prepare input & dummy output: {(t1 - tic)*1000} ms, update attn mask: {(toc - t4)*1000} ms, "
                  f"argmax: {(t3 - t2)*1000} ms, cat input id: {(t4 - t3) * 1000} ms, logtis processor: {(t2 - t1)*1000} ms"
                  f" total generate: {(toc - t1)*1000} ms")

        # Stop on eos and remove content after eos
        if eos_token_id_set is not None:
            output_ids_list = output_ids[0].tolist()
            first_eos_idx = -1
            for out_idx, out_id in enumerate(output_ids_list):
                if out_id in eos_token_id_set:
                    first_eos_idx = out_idx
                    break
            if first_eos_idx > -1:
                if streamer is not None:
                    streamer.put(output_ids[:(first_eos_idx + 1)].cpu())
                step -= (len(output_ids_list) - first_eos_idx - 1)
                break
        if streamer is not None:
            streamer.put(output_ids.cpu())

    step = min(step, max_new_tokens)
    self.n_token_generated = step

    print(f"=========First token cost {self.first_token_time:.4f} s=========")
    if len(self.last_token_time) > 1:
        self.first_cost = self.first_token_time
        self.rest_cost_mean = np.mean(self.last_token_time)
        print(f"=========Rest tokens cost average {self.rest_cost_mean:.4f} s "
              f"({len(self.last_token_time)} tokens in all)=========")

    if streamer is not None:
        streamer.end()

    return input_ids[:, : input_len + step]
