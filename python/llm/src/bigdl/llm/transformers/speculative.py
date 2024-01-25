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
                    'auto_th_stop_draft', 'auto_parameters', 'repetition_penalty']:
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


def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50,
           top_p: float=0.7, temperature: float=0.7):

    if return_probs:
        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature,
                                            top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1),
                                           num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
        return output_ids, probs
    else:
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature,
                                            top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1),
                                           num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
        return output_ids


def clear_benchmarks(self):
    self.first_token_time = 0
    self.generate_time = []
    self.draft_time = []
    self.verify_time = []
    self.draft_num = []
    self.accept_num = []
    self.n_drafted = 0
    self.n_matched = 0


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
    past_key_values1 = []

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
                          return_dict=True,
                          use_cache=True)
            logits = output['logits']
            logits = logits[:, -1:]
            logits[:, -1, :] = logits_processor(current_input_ids, logits[:, -1, :])
            output_ids = sample(logits, do_sample=generation_config.do_sample,
                                top_k=generation_config.top_k, top_p=generation_config.top_p,
                                temperature=generation_config.temperature)
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

            # init draft_self_past_key_values:past_key_values1 and assign initial fp32 value
            if self.device.type == 'cpu' and step == 1:
                for i in range(len(past_key_values)):
                    len0 = past_key_values[i][0].size(0)
                    len1 = past_key_values[i][0].size(1)
                    len2 = past_key_values[i][0].size(2)
                    len3 = past_key_values[i][0].size(3)
                    if self.config.model_type == "qwen":
                        k0 = torch.ones(len0, len2, len1 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        v0 = torch.ones(len0, len2, len1 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        k0 = k0.transpose(1, 2)
                        v0 = v0.transpose(1, 2)
                        past_key_values1.append((k0, v0))
                        past_key_values1[i][0][:, :len1, :, :] = past_key_values[i][0].to(
                            torch.float32)
                        past_key_values1[i][1][:, :len1, :, :] = past_key_values[i][1].to(
                            torch.float32)
                    elif self.config.model_type == "chatglm":
                        k0 = torch.ones(len1, len2, len0 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        v0 = torch.ones(len1, len2, len0 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        k0 = k0.permute(2, 0, 1, 3)
                        v0 = v0.permute(2, 0, 1, 3)
                        past_key_values1.append((k0, v0))
                        past_key_values1[i][0][:len0, :, :, :] = past_key_values[i][0].to(
                            torch.float32)
                        past_key_values1[i][1][:len0, :, :, :] = past_key_values[i][1].to(
                            torch.float32)
                    else:
                        k0 = torch.ones(len0, len1, len2 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        v0 = torch.ones(len0, len1, len2 + max_new_tokens, len3,
                                        dtype=torch.float32)
                        past_key_values1.append((k0, v0))
                        past_key_values1[i][0][:, :, :len2, :] = past_key_values[i][0].to(
                            torch.float32)
                        past_key_values1[i][1][:, :, :len2, :] = past_key_values[i][1].to(
                            torch.float32)

            # each iter cut off cur_len kv_cache from past_key_values1
            if self.device.type == 'cpu':
                tmp_past_key_values = []
                for i in range(len(past_key_values)):
                    if self.config.model_type == "qwen":
                        len1 = past_key_values[0][0].size(1)
                        k0 = past_key_values1[i][0][:, :len1, :, :]
                        v0 = past_key_values1[i][1][:, :len1, :, :]
                        tmp_past_key_values.append((k0, v0))
                    elif self.config.model_type == "chatglm":
                        len0 = past_key_values[0][0].size(0)
                        k0 = past_key_values1[i][0][:len0, :, :, :]
                        v0 = past_key_values1[i][1][:len0, :, :, :]
                        tmp_past_key_values.append((k0, v0))
                    else:
                        len2 = past_key_values[0][0].size(2)
                        k0 = past_key_values1[i][0][:, :, :len2, :]
                        v0 = past_key_values1[i][1][:, :, :len2, :]
                        tmp_past_key_values.append((k0, v0))
                draft_past_key_values = tmp_past_key_values
            else:
                draft_past_key_values = past_key_values
            draft_generate_ids[:, 0] = current_input_ids
            tic = time.time()
            # Draft model auto-regressively generate k tokens
            # Early stop when prob less then th_stop_draft
            for step_draft in range(max_step_draft):
                if self.config.model_type == "chatglm":
                    past_key_value_len = past_key_values[0][0].shape[0]
                    position_ids = torch.Tensor([[past_key_value_len + step_draft]]).long()
                    draft_output = draft_model(input_ids=draft_current_input_ids,
                                               past_key_values=draft_past_key_values,
                                               return_dict=True,
                                               use_cache=True,
                                               position_ids=position_ids)
                else:
                    draft_output = draft_model(input_ids=draft_current_input_ids,
                                               past_key_values=draft_past_key_values,
                                               return_dict=True,
                                               use_cache=True)
                temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                            draft_generate_ids[:, 1:step_draft+1]), dim=-1)
                logits = draft_output['logits']
                logits[:, -1, :] = logits_processor(temp_input_ids,
                                                    draft_output['logits'][:, -1, :])
                draft_output_ids, draft_output_probs = sample(
                    logits,
                    return_probs=True,
                    do_sample=generation_config.do_sample,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    temperature=generation_config.temperature)
                draft_generate_ids[:, step_draft+1] = draft_output_ids
                draft_current_input_ids = draft_output_ids
                draft_past_key_values = draft_output['past_key_values']
                # check if draft prob is less then th_stop_draft
                # Draft number + step >= max output token number
                if draft_output_probs.item() < th_stop_draft or \
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
            if self.config.model_type == "chatglm":
                past_key_value_len = past_key_values[0][0].shape[0]
                position_ids = torch.arange(drafted_input_ids.shape[1], dtype=torch.long,
                                            device=drafted_input_ids.device)
                position_ids = position_ids.unsqueeze(0).repeat(1, 1) + past_key_value_len
                output = self(input_ids=drafted_input_ids,
                              past_key_values=past_key_values,
                              return_dict=True,
                              use_cache=True,
                              position_ids=position_ids)
            else:
                output = self(input_ids=drafted_input_ids,
                              past_key_values=past_key_values,
                              return_dict=True,
                              use_cache=True)
            logits = output['logits']
            temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                        draft_generate_ids[:, 1:step_draft + 2]), dim=-1)
            for i in range(logits.size(1)):
                logits[:, i, :] = logits_processor(temp_input_ids[:, :input_ids.size(1)+step+i],
                                                   output['logits'][:, i, :])
            output_ids = sample(logits, do_sample=generation_config.do_sample,
                                top_k=generation_config.top_k, top_p=generation_config.top_p,
                                temperature=generation_config.temperature)
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.verify_time.append(toc - tic)
            self.generate_time.append(self.draft_time[-1] + self.verify_time[-1])

            past_key_values = output['past_key_values']
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
                if self.config.model_type == "qwen":
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
                         v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
                    ]

            # Each iter assign new_matched kv_cache to past_key_values1
            if self.device.type == 'cpu':
                for i in range(len(past_key_values)):
                    if self.config.model_type == "qwen":
                        size = tmp_past_key_values[i][0].size(1)
                        size1 = past_key_values[i][0].size(1)
                        past_key_values1[i][0][:, size:size1, :, :] = \
                            past_key_values[i][0][:, size:size1, :, :].to(torch.float32)
                        past_key_values1[i][1][:, size:size1, :, :] = \
                            past_key_values[i][1][:, size:size1, :, :].to(torch.float32)
                    elif self.config.model_type == "chatglm":
                        size = tmp_past_key_values[i][0].size(0)
                        size1 = past_key_values[i][0].size(0)
                        past_key_values1[i][0][size:size1, :, :, :] = \
                            past_key_values[i][0][size:size1, :, :, :].to(torch.float32)
                        past_key_values1[i][1][size:size1, :, :, :] = \
                            past_key_values[i][1][size:size1, :, :, :].to(torch.float32)
                    else:
                        size = tmp_past_key_values[i][0].size(2)
                        size1 = past_key_values[i][0].size(2)
                        past_key_values1[i][0][:, :, size:size1, :] = \
                            past_key_values[i][0][:, :, size:size1, :].to(torch.float32)
                        past_key_values1[i][1][:, :, size:size1, :] = \
                            past_key_values[i][1][:, :, size:size1, :].to(torch.float32)

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
