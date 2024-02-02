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
# https://github.com/dilab-zju/self-speculative-decoding/blob/main/decoding.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation
# /utils.py
#

import torch
import time
import os
import copy
import logging
import warnings
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import top_k_top_p_filtering, GenerationConfig, \
    LogitsProcessorList, StoppingCriteriaList
from bigdl.llm.utils.common import invalidInputError

# patch GenerationMixin.generate
from transformers import GenerationMixin
original_generate = GenerationMixin.generate

logger = logging.getLogger("bigdl.llm.speculative")


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
    if hasattr(self, "draft_model"):
        # do speculative decoding
        # TODO: maybe add other way to double check
        new_speculative_kwargs = {}
        for var in ['max_new_tokens', 'max_step_draft', 'th_stop_draft', 'do_sample',
                    'top_k', 'top_p', 'temperature', 'hf_adjust',
                    'auto_th_stop_draft', 'auto_parameters', 'repetition_penalty',
                    'attention_mask']:
            value = kwargs.pop(var, None)
            if value is not None:
                new_speculative_kwargs[var] = value
        return self.speculative_generate(inputs=inputs,
                                         draft_model=self.draft_model,
                                         **new_speculative_kwargs)
    else:
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


def greedy(logits, return_probs: bool=False):
    if return_probs:
        all_probs = logits.softmax(-1)
        probs, output_ids = torch.max(all_probs, dim=-1)
        return output_ids, probs
    else:
        output_ids = torch.argmax(logits, dim=-1)
        return output_ids


def deepmind_sample(logits, return_probs: bool=False, top_k: int=50,
                    top_p: float=0.7, temperature: float=0.7):
    prob_list = logits_to_probs(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    output_ids = multinomial_sample_one_no_sync(prob_list)
    if return_probs:
        all_probs = logits.softmax(-1)
        probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        return output_ids, prob_list, probs
    else:
        return output_ids, prob_list


def logits_to_probs(logits, top_k: int=50, top_p: float=0.7, temperature: float=0.7):
    invalidInputError(top_k != 1 and top_p != 0.0 and temperature != 0.0,
                      "top_k != 1 and top_p != 0.0 and temperature != 0.0 if do_sample=True")
    _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature,
                                    top_k=top_k, top_p=top_p)
    prob_list = _logits.softmax(-1)

    return prob_list


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int64)


def clear_benchmarks(self):
    self.first_token_time = 0
    self.generate_time = []
    self.draft_time = []
    self.verify_time = []
    self.draft_num = []
    self.accept_num = []
    self.n_drafted = 0
    self.n_matched = 0


def _prepare_past_key_values_storage_cpu(self, past_key_values,
                                         max_new_tokens, _enable_ipex=False):
    past_key_values_storage = []
    if _enable_ipex:
        ipex_past_key_values = []
        cur_len = past_key_values[0][0].size(1)
        ipex_past_key_values = [
            [pkv[1].permute(1, 2, 0, 3)[:, :, :cur_len, :],
                pkv[2].permute(1, 2, 0, 3)[:, :, :cur_len, :]]
            for pkv in past_key_values
        ]

    for i in range(len(past_key_values)):
        if not _enable_ipex:
            len0 = past_key_values[i][0].size(0)
            len1 = past_key_values[i][0].size(1)
            len2 = past_key_values[i][0].size(2)
            len3 = past_key_values[i][0].size(3)
        else:
            len0 = past_key_values[i][1].size(1)
            len1 = past_key_values[i][1].size(2)
            len2 = past_key_values[i][0].size(2)  # seq length
            len3 = past_key_values[i][1].size(3)
        if self.config.model_type == "qwen":
            k0 = torch.ones(len0, len2, len1 + max_new_tokens, len3,
                            dtype=torch.float32)
            v0 = torch.ones(len0, len2, len1 + max_new_tokens, len3,
                            dtype=torch.float32)
            k0 = k0.transpose(1, 2)
            v0 = v0.transpose(1, 2)
            past_key_values_storage.append((k0, v0))
            past_key_values_storage[i][0][:, :len1, :, :] = past_key_values[i][0].to(
                torch.float32)
            past_key_values_storage[i][1][:, :len1, :, :] = past_key_values[i][1].to(
                torch.float32)
        elif self.config.model_type == "chatglm":
            k0 = torch.ones(len1, len2, len0 + max_new_tokens, len3,
                            dtype=torch.float32)
            v0 = torch.ones(len1, len2, len0 + max_new_tokens, len3,
                            dtype=torch.float32)
            k0 = k0.permute(2, 0, 1, 3)
            v0 = v0.permute(2, 0, 1, 3)
            past_key_values_storage.append((k0, v0))
            past_key_values_storage[i][0][:len0, :, :, :] = past_key_values[i][0].to(
                torch.float32)
            past_key_values_storage[i][1][:len0, :, :, :] = past_key_values[i][1].to(
                torch.float32)
        else:
            k0 = torch.ones(len0, len1, len2 + max_new_tokens, len3,
                            dtype=torch.float32)
            v0 = torch.ones(len0, len1, len2 + max_new_tokens, len3,
                            dtype=torch.float32)
            past_key_values_storage.append((k0, v0))
            if not _enable_ipex:
                past_key_values_storage[i][0][:, :, :len2, :] = past_key_values[i][0].to(
                    torch.float32)
                past_key_values_storage[i][1][:, :, :len2, :] = past_key_values[i][1].to(
                    torch.float32)
            else:
                past_key_values_storage[i][0][:, :, :len2, :] = ipex_past_key_values[i][0].to(
                    torch.float32)
                past_key_values_storage[i][1][:, :, :len2, :] = ipex_past_key_values[i][1].to(
                    torch.float32)

    return past_key_values_storage


def _prepare_draft_past_key_values_cpu(self, past_key_values, past_key_values_storage):
    tmp_past_key_values = []
    for i in range(len(past_key_values)):
        if self.config.model_type == "qwen":
            len1 = past_key_values[0][0].size(1)
            k0 = past_key_values_storage[i][0][:, :len1, :, :]
            v0 = past_key_values_storage[i][1][:, :len1, :, :]
            tmp_past_key_values.append((k0, v0))
        elif self.config.model_type == "chatglm":
            len0 = past_key_values[0][0].size(0)
            k0 = past_key_values_storage[i][0][:len0, :, :, :]
            v0 = past_key_values_storage[i][1][:len0, :, :, :]
            tmp_past_key_values.append((k0, v0))
        else:
            len2 = past_key_values[0][0].size(2)
            k0 = past_key_values_storage[i][0][:, :, :len2, :]
            v0 = past_key_values_storage[i][1][:, :, :len2, :]
            tmp_past_key_values.append((k0, v0))
    return tmp_past_key_values


def _update_past_key_values_storage_cpu(self, past_key_values, past_key_values_storage,
                                        original_draft_past_key_values, _enable_ipex=False):
    for i in range(len(past_key_values)):
        if not _enable_ipex:
            if self.config.model_type == "qwen":
                size = original_draft_past_key_values[i][0].size(1)
                size1 = past_key_values[i][0].size(1)
                past_key_values_storage[i][0][:, size:size1, :, :] = \
                    past_key_values[i][0][:, size:size1, :, :].to(torch.float32)
                past_key_values_storage[i][1][:, size:size1, :, :] = \
                    past_key_values[i][1][:, size:size1, :, :].to(torch.float32)
            elif self.config.model_type == "chatglm":
                size = original_draft_past_key_values[i][0].size(0)
                size1 = past_key_values[i][0].size(0)
                past_key_values_storage[i][0][size:size1, :, :, :] = \
                    past_key_values[i][0][size:size1, :, :, :].to(torch.float32)
                past_key_values_storage[i][1][size:size1, :, :, :] = \
                    past_key_values[i][1][size:size1, :, :, :].to(torch.float32)
            else:
                size = original_draft_past_key_values[i][0].size(2)
                size1 = past_key_values[i][0].size(2)
                past_key_values_storage[i][0][:, :, size:size1, :] = \
                    past_key_values[i][0][:, :, size:size1, :].to(torch.float32)
                past_key_values_storage[i][1][:, :, size:size1, :] = \
                    past_key_values[i][1][:, :, size:size1, :].to(torch.float32)
        else:
            size = original_draft_past_key_values[i][0].size(2)
            size1 = past_key_values[i][0].size(1)
            delta_past_key = \
                past_key_values[i][1][size:size1, :, :, :].permute(1, 2, 0, 3)
            delta_past_value = \
                past_key_values[i][2][size:size1, :, :, :].permute(1, 2, 0, 3)

            past_key_values_storage[i][0][:, :, size:size1, :] = \
                delta_past_key.to(torch.float32)
            past_key_values_storage[i][1][:, :, size:size1, :] = \
                delta_past_value.to(torch.float32)


@torch.no_grad()
def speculative_generate(self,
                         inputs: Optional[torch.Tensor] = None,
                         draft_model=None,
                         max_new_tokens=10,
                         max_step_draft=8,
                         th_stop_draft=0.8,
                         auto_th_stop_draft=True,
                         auto_parameters=[1, 0.5, 0.9, 1e-2, 0.9],
                         hf_adjust=False,
                         generation_config: Optional[GenerationConfig] = None,
                         attention_mask=None,
                         **sampling_kwargs):
    invalidInputError(draft_model is not None,
                      "Draft model should be provided.")

    if generation_config is None:
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    # All unused kwargs must be model kwargs
    model_kwargs = generation_config.update(**sampling_kwargs)
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, "
                "you may observe unexpected behavior. Please pass your input's "
                "`attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:"
                       f"{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 2. Set generation parameters if not already defined
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    # Removed not used

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work,
        # because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding "
                "was detected! For correct generation results, please set "
                "`padding_side='left'` when initializing the tokenizer."
            )
    else:
        invalidInputError(False, "encoder-decoder models are not supported now.")

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # if streamer is not None:
    #     streamer.put(input_ids.cpu())

    input_ids_length = input_ids.shape[-1]

    # Here we use sample generation mode
    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )

    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    step = 0
    step_draft = 0
    step_verify = 0

    draft_gen_length = max_step_draft + 6 if hf_adjust else max_step_draft + 1
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft],
                               dtype=torch.long, device=self.device)
    draft_generate_ids = torch.empty([input_ids.size(0), draft_gen_length],
                                     dtype=torch.long, device=self.device)
    past_key_values = None
    past_key_values_storage = []

    _enable_ipex = os.getenv("BIGDL_OPT_IPEX")
    _enable_ipex = (_enable_ipex is not None) and (_enable_ipex.lower() == "true")
    if _enable_ipex:
        if not ((self.config.model_type == 'baichuan' and self.config.hidden_size == 5120) or
                ('llama' in self.config.model_type) or
                ("mistral" in self.config.model_type)):
            invalidInputError(False, "BigDL Speculative Decoding with IPEX BF16 only supports \
                                      Llama, Baichuan2-13b and Mistral models currently.")

    tmp_matchness = 0
    e2e_tic = 0.0

    self.clear_benchmarks()

    # Example:
    # Target model forward for the first token
    # Step 1. target_model(prompt) -> a
    # Generate k drafts, k = 3
    # Step 2. draft_model(a) -> b, c, d
    # Verify k drafts -> k + 1 results (f is always accepted)
    # Step 3. target_model (a, b, c, d) -> b, c, e, f
    # Compare drafts with results
    # Step 4. (b, c, e) match (b, c, d) -> b, c
    # Final, f will be the next input, just like a
    # Step 5. Final-> b, c, f
    while True:
        if step >= max_new_tokens:
            break

        if step == 0:
            # first token use full model
            tic = time.time()
            output = self(input_ids=current_input_ids,
                          past_key_values=past_key_values,
                          attention_mask=attention_mask,
                          return_dict=True,
                          use_cache=True)
            logits = output['logits']
            logits = logits[:, -1:]
            logits[:, -1, :] = logits_processor(current_input_ids, logits[:, -1, :])
            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
            else:
                output_ids = greedy(logits)
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output['past_key_values']
            step += 1
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.first_token_time = toc - tic
            e2e_tic = time.time()
        else:
            draft_current_input_ids = current_input_ids
            # Target model KV cache to draft model

            if self.device.type == 'cpu':
                # init past_key_values_storage and assign initial fp32 value
                if step == 1:
                    past_key_values_storage = \
                        _prepare_past_key_values_storage_cpu(self, past_key_values,
                                                             max_new_tokens, _enable_ipex)
                # each iter cut off cur_len kv_cache from past_key_values1
                draft_past_key_values = \
                    _prepare_draft_past_key_values_cpu(self, past_key_values,
                                                       past_key_values_storage)
                original_draft_past_key_values = draft_past_key_values
            else:
                draft_past_key_values = past_key_values
            draft_generate_ids[:, 0] = current_input_ids
            draft_prob_list = []
            tic = time.time()
            random_probs = None
            if generation_config.do_sample:
                random_probs = torch.rand(max_step_draft, device=self.device, dtype=self.dtype)
            # Draft model auto-regressively generate k tokens
            # Early stop when prob less then th_stop_draft
            for step_draft in range(max_step_draft):
                if attention_mask is None:
                    draft_attention_mask = None
                else:
                    appended_len = step_draft + step
                    ones_to_append = torch.ones(attention_mask.size(0), appended_len)
                    draft_attention_mask = torch.cat((attention_mask, ones_to_append), dim=1)
                forward_args = {
                    "input_ids": draft_current_input_ids,
                    "past_key_values": draft_past_key_values,
                    "attention_mask": draft_attention_mask,
                    "return_dict": True,
                    "use_cache": True,
                }
                if self.config.model_type == "chatglm":
                    past_key_value_len = past_key_values[0][0].shape[0]
                    position_ids = torch.Tensor([[past_key_value_len + step_draft]]).long()
                    forward_args["position_ids"] = position_ids
                elif self.config.model_type == "gptj":
                    past_length = draft_past_key_values[0][0].size(1)
                    position_ids = torch.Tensor([[past_length]]).long().to(self.device)
                    forward_args["position_ids"] = position_ids
                draft_output = draft_model(**forward_args)
                temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                            draft_generate_ids[:, 1:step_draft+1]), dim=-1)
                logits = draft_output['logits']
                logits[:, -1, :] = logits_processor(temp_input_ids,
                                                    draft_output['logits'][:, -1, :])
                if generation_config.do_sample:
                    draft_output_ids, draft_probs, draft_output_probs = deepmind_sample(
                        logits,
                        return_probs=True,
                        top_k=generation_config.top_k,
                        top_p=generation_config.top_p,
                        temperature=generation_config.temperature)
                    draft_prob_list.append(draft_probs)
                else:
                    draft_output_ids, draft_output_probs = greedy(
                        logits,
                        return_probs=True)
                draft_generate_ids[:, step_draft+1] = draft_output_ids
                draft_current_input_ids = draft_output_ids
                draft_past_key_values = draft_output['past_key_values']
                # check if draft prob is less then th_stop_draft
                # Draft number + step >= max output token number
                th_random = 1 if random_probs is None else random_probs[step_draft]
                if (draft_output_probs.item() < th_stop_draft and th_random > 0.3) or \
                        step + step_draft + 2 >= max_new_tokens:
                    break
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.draft_time.append(toc - tic)
            drafted_n_tokens = step_draft + 1
            # raft input + raft completion
            drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1]
            self.draft_num.append(drafted_n_tokens)
            tic = time.time()
            # Target model verify drafts
            # input.size is k + 1, 1 previous token + k drafts
            # verified output.size is k + 1, k token + 1 final
            # Final token is always accepted
            if attention_mask is None:
                cur_attention_mask = None
            else:
                appended_len = drafted_input_ids.size(1) + step - 1
                ones_to_append = torch.ones(attention_mask.size(0), appended_len)
                cur_attention_mask = torch.cat((attention_mask, ones_to_append), dim=1)
            if _enable_ipex and hasattr(self, "trace_graph"):
                if self.config.model_type == "baichuan":
                    output = self.trace_graph(input_ids=drafted_input_ids,
                                              attention_mask=cur_attention_mask,
                                              past_key_values=past_key_values,
                                              )
                elif "llama" in self.config.model_type:
                    past_key_value_len = past_key_values[0][0].shape[2]
                    position_ids = torch.arange(drafted_input_ids.shape[1], dtype=torch.long,
                                                device=drafted_input_ids.device).unsqueeze(0)
                    position_ids = position_ids.repeat(1, 1) + past_key_value_len
                    output = self.trace_graph(input_ids=drafted_input_ids,
                                              attention_mask=cur_attention_mask,
                                              position_ids=position_ids,
                                              past_key_values=past_key_values,
                                              )
                elif "mistral" in self.config.model_type:
                    past_key_value_len = past_key_values[0][0].shape[2]
                    seq_len = drafted_input_ids.shape[1]
                    position_ids = torch.arange(past_key_value_len,
                                                seq_len + past_key_value_len,
                                                dtype=torch.long,
                                                device=drafted_input_ids.device)
                    position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                    output = self.trace_graph(input_ids=drafted_input_ids,
                                              attention_mask=cur_attention_mask,
                                              past_key_values=past_key_values,
                                              position_ids=position_ids,
                                              )
                logits = output[0]
                past_key_values = output[1]
            else:
                forward_args = {
                    "input_ids": drafted_input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": cur_attention_mask,
                    "return_dict": True,
                    "use_cache": True,
                }
                if self.config.model_type == "chatglm":
                    past_key_value_len = past_key_values[0][0].shape[0]
                    position_ids = torch.arange(drafted_input_ids.shape[1], dtype=torch.long,
                                                device=drafted_input_ids.device)
                    position_ids = position_ids.unsqueeze(0).repeat(1, 1) + past_key_value_len
                    forward_args["position_ids"] = position_ids
                elif self.config.model_type == "gptj":
                    past_length = past_key_values[0][0].size(1)
                    input_len = drafted_input_ids.shape[1]
                    position_ids = torch.arange(past_length, input_len + past_length,
                                                dtype=torch.long, device=drafted_input_ids.device)
                    position_ids = position_ids.unsqueeze(0).view(-1, input_len)
                    forward_args["position_ids"] = position_ids
                output = self(**forward_args)
            if isinstance(output, dict):
                logits = output['logits']
                past_key_values = output['past_key_values']
            temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                        draft_generate_ids[:, 1:step_draft + 2]), dim=-1)
            for i in range(logits.size(1)):
                logits[:, i, :] = logits_processor(temp_input_ids[:, :input_ids.size(1)+step+i],
                                                   logits[:, i, :])
            if generation_config.do_sample:
                target_probs = logits_to_probs(logits,
                                               top_k=generation_config.top_k,
                                               top_p=generation_config.top_p,
                                               temperature=generation_config.temperature)
            else:
                output_ids = greedy(logits)
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.verify_time.append(toc - tic)
            self.generate_time.append(self.draft_time[-1] + self.verify_time[-1])

            past_key_values = output['past_key_values']

            if generation_config.do_sample:
                draft_tokens = drafted_input_ids[:, 1:].squeeze(0)
                draft_probs = torch.stack(draft_prob_list).squeeze((1, 2))

                # q: target prob, p: draft prob
                # q >= p: always accept draft token
                # q < p: q/p prob to accept draft token
                p = draft_probs[torch.arange(0, drafted_n_tokens), draft_tokens]
                q = target_probs[torch.arange(0, drafted_n_tokens), draft_tokens]
                accept_draft_prob = torch.minimum(torch.ones(()), q[:drafted_n_tokens] / p)
                rejected_locations = (random_probs[:drafted_n_tokens] > accept_draft_prob).nonzero()

                if rejected_locations.shape[0] == 0:    # All draft tokens have been accepted
                    max_matched = drafted_n_tokens + 1
                    last_token = multinomial_sample_one_no_sync(target_probs[-1])
                    output_ids = torch.cat([draft_tokens, last_token])
                else:
                    max_matched = rejected_locations[0].item()
                    p = draft_probs[max_matched]
                    q = target_probs[max_matched]
                    resample_prob = q - p
                    resample_prob = torch.where(resample_prob > 0, resample_prob, 0.0)
                    resample_prob = resample_prob / resample_prob.sum()
                    next_token = multinomial_sample_one_no_sync(resample_prob)
                    output_ids = torch.cat([draft_tokens[:max_matched], next_token])
                    max_matched += 1
                output_ids = output_ids.unsqueeze(0)
            else:
                # Compare drafts with target verified outputs
                # Drafts start from [1, k]
                # Verified output start from [0, k - 1]
                # including the one generated by the base model
                max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0)
                max_matched = max_matched.sum(-1).item() + 1

            max_of_max_matched = output_ids.size(1)
            # Accept number is max_matched, min is 1
            self.accept_num.append(max_matched)
            # Clean up target model KV cache
            if max_of_max_matched != max_matched:
                output_ids = output_ids[:, :max_matched]
                # For Qwen
                if _enable_ipex:
                    cur_len = past_key_values[0][0].size(1)
                    delta = max_of_max_matched - max_matched
                    tmp = torch.empty(1, (cur_len - delta), (cur_len - delta), 1,
                                      dtype=torch.long,
                                      ).contiguous()
                    past_key_values = [[tmp, key_cache, value_cache, beam_idx]
                                       for _, key_cache, value_cache, beam_idx in past_key_values]
                else:
                    if self.config.model_type in ["qwen", "gptj"]:
                        past_key_values = [
                            (k[:, :-(max_of_max_matched - max_matched), :],
                             v[:, :-(max_of_max_matched - max_matched), :])
                            for k, v in past_key_values
                        ]
                    elif self.config.model_type == "chatglm":
                        # for chatglm, cache shape is [sl, bs, nh, hn]
                        past_key_values = [
                            (k[:-(max_of_max_matched - max_matched), :, :, :],
                             v[:-(max_of_max_matched - max_matched), :, :, :])
                            for k, v in past_key_values
                        ]
                    elif self.config.model_type == "baichuan":
                        past_key_values = [
                            (k[:, :, :-(max_of_max_matched - max_matched), :],
                             v[:, :, :-(max_of_max_matched - max_matched), :])
                            for k, v in past_key_values
                        ]
                    else:
                        past_key_values = [
                            (k[:, :, :-(max_of_max_matched - max_matched)],
                             v[:, :, :-(max_of_max_matched - max_matched)])
                            for k, v in past_key_values
                        ]

            # Each iter assign new_matched kv_cache to past_key_values1
            if self.device.type == 'cpu':
                _update_past_key_values_storage_cpu(self, past_key_values, past_key_values_storage,
                                                    original_draft_past_key_values,
                                                    _enable_ipex)

            generate_ids[:, step:step+output_ids.size(1)] = output_ids
            current_input_ids = output_ids[:, -1:]

            step += output_ids.size(1)

            # remove one generated by the base model
            self.n_matched += max_matched - 1
            self.n_drafted += drafted_n_tokens
            step_verify += 1

            if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                tmp_matchness = auto_parameters[1]*(tmp_matchness) + \
                    (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                if tmp_matchness < auto_parameters[2]:
                    new_th_stop_draft = th_stop_draft+auto_parameters[3]
                else:
                    if drafted_n_tokens == max_step_draft:
                        new_th_stop_draft = th_stop_draft
                    else:
                        new_th_stop_draft = th_stop_draft - auto_parameters[3]
                th_stop_draft = auto_parameters[4] * th_stop_draft + \
                    (1-auto_parameters[4]) * new_th_stop_draft

            if hf_adjust:
                if (max_matched - 1) == max_step_draft:
                    max_step_draft = min(draft_gen_length - 1, max_step_draft + 1)
                else:
                    max_step_draft = max(1, max_step_draft - 1)

        # Stop on eos and remove content after eos
        output_ids_list = output_ids[0].tolist()
        if generation_config.eos_token_id in output_ids_list:
            idx = output_ids_list.index(generation_config.eos_token_id)
            step -= (len(output_ids_list) - idx - 1)
            break

    step = min(step, max_new_tokens)
    e2e_toc = time.time()
    self.n_token_generated = step
    self.e2e_time_without_first = e2e_toc - e2e_tic

    generate_ids = torch.cat([input_ids, generate_ids[:, :step]], dim=-1)

    return generate_ids
