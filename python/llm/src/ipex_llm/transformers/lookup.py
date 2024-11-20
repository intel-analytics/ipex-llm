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
# /candidate_generator.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation
# /utils.py
#

from typing import Callable, List, Optional, Tuple
import os
import torch
import time
import copy
import random
import logging
import transformers
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ipex_llm.transformers.speculative import greedy, deepmind_sample, logits_to_probs,\
    _crop_past_key_values, _prepare_generate_args, _non_cpu_ipex_verify, clear_benchmarks,\
    _prepare_generate_args_4_45
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.utils import get_xpu_device_type

logger = logging.getLogger("ipex_llm.lookup")

# patch GenerationMixin.generate
from transformers import GenerationMixin
original_generate = GenerationMixin.generate
query_group_size = 16

# may tune it with more tested data
PERFORMANCE_MODE_LOOKUP_INPUT_THRESHOLD = 100


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
    lookahead = kwargs.pop("lookahead", None)
    perf_mode = os.environ.get("IPEX_LLM_PERFORMANCE_MODE", None)

    input_tensor_shape = None
    is_inputs_embeds = False
    if inputs is not None:
        input_tensor_shape = inputs.shape
    else:
        input_ids = kwargs.get("input_ids", None)
        if input_ids is not None:
            input_tensor_shape = input_ids.shape
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is not None:
                is_inputs_embeds = True
                input_tensor_shape = inputs_embeds.shape

    if perf_mode == "1" and lookahead is None:
        if input_tensor_shape is not None and \
                input_tensor_shape[1] >= PERFORMANCE_MODE_LOOKUP_INPUT_THRESHOLD \
                and not is_inputs_embeds:
            lookahead = 2  # default to 2 now

    if lookahead:
        from ipex_llm.transformers.convert import get_enable_ipex
        _enable_ipex = get_enable_ipex()

        if self.device.type == "cpu" and _enable_ipex:
            logger.warning("Prompt lookup is currently not supported on CPU with IPEX, "
                           "fallback to original generate.")
            kwargs.pop("max_matching_ngram_size", None)
        elif input_tensor_shape is not None and input_tensor_shape[0] > 1:
            logger.warning("Prompt lookup is currently not supported with batch inference, "
                           "fallback to original generate.")
            kwargs.pop("max_matching_ngram_size", None)
        elif kwargs.get("num_beams", None) not in [None, 1]:
            logger.warning("Prompt lookup is currently not supported with num_beams != 1, "
                           "fallback to original generate.")
            kwargs.pop("max_matching_ngram_size", None)
        else:
            # Do prompt lookup generation
            # If lookahead is provided, we will use lookup_generate instead of
            #  spec_generate, remove vars for spec_generate and warn the user
            spec_params = []
            for var in ['max_step_draft', 'th_stop_draft', 'hf_adjust',
                        'auto_th_stop_draft', 'auto_parameters', 'min_step_draft',
                        'th_batch_num']:
                value = kwargs.pop(var, None)
                if value is not None:
                    spec_params.append(var)
            if len(spec_params) > 0:
                logger.warning("Since you call the generate with lookahead parameter, "
                               f"Speculative decoding parameters {spec_params} are "
                               "removed in the generation.")
            return self.lookup_generate(inputs=inputs,
                                        num_output_tokens=lookahead,
                                        generation_config=generation_config,
                                        streamer=streamer,
                                        logits_processor=logits_processor,
                                        stopping_criteria=stopping_criteria,
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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


def tensor2key(key_tensor: torch.LongTensor):
    return tuple(key_tensor.tolist())


# This class is copied from https://github.com/huggingface/transformers/blob/main/src
# /transformers/generation/candidate_generator.py
class PromptLookupCandidateGenerator():
    """
    `CandidateGenerator` class to be used for prompt lookup generation.
    This class generates candidates
    by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information:
    https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        max_matching_ngram_size (`int`):
            The maximum ngram size to be considered for matching in the prompt
        num_output_tokens (`int`):
            The number of tokens to be output as candidate tokens.
    """

    def __init__(
        self,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        device: str = "arc",
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2

        if device in ["mtl", "lnl"]:
            self.max_candidates = 3
            self.min_candidates = 0
        else:
            self.max_candidates = 9
            self.min_candidates = 0

        self.lookup_table = {}
        invalidInputError(self.max_matching_ngram_size > 0 and self.num_output_tokens > 0,
                          "Invalid max_matching_ngram_size or num_output_tokens")

    def init_look_up_table(self,
                           input_ids: torch.LongTensor):
        for ngram_size in range(min(self.max_matching_ngram_size, input_ids.shape[1]), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.cpu().unfold(dimension=1, size=ngram_size, step=1)
            for idx in range(windows.size(1)):
                window = tensor2key(windows[0, idx])
                if window not in self.lookup_table:
                    self.lookup_table[window] = idx

    def update_look_up_table(self,
                             new_input_ids: torch.LongTensor):
        # Maintain a look up table
        window = tensor2key(new_input_ids[0, -self.max_matching_ngram_size:])
        for ngram_size in range(self.max_matching_ngram_size):
            if window[ngram_size:] not in self.lookup_table:
                self.lookup_table[window[ngram_size:]] = \
                    new_input_ids.size(1)-self.max_matching_ngram_size+ngram_size

    def get_n_gram_idx(self,
                       ngram_tensor: torch.LongTensor):
        key = tensor2key(ngram_tensor)
        return self.lookup_table[key]

    def get_candidates(self,
                       input_ids: torch.LongTensor)-> Tuple[torch.LongTensor,
                                                            Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(num_candidates, candidate_length)`:
            The candidate sequences to be tried.
        """
        if self.num_output_tokens == 0:
            return input_ids, None
        input_length = input_ids.size(1)

        chosen_ids = None
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # # Get the indices of matches
            idx = self.get_n_gram_idx(ngram_tensor)

            # Iterate through match indices to find a valid continuation
            start_idx = idx + ngram_size
            end_idx = start_idx + self.num_output_tokens
            end_idx = min(end_idx, input_length)

            if start_idx < end_idx:
                chosen_ids = input_ids[0, start_idx:end_idx]
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged,
            # reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here,
        # so returning None
        return candidate_input_ids, None

    def update_candidate_strategy(self, candidate_num: int, num_matches: int, accept_rate: float):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        if self.num_output_tokens == 0:
            ran = random.random() - 0.15
            if ran <= accept_rate:
                self.num_output_tokens = 1
        elif num_matches == self.num_output_tokens:
            self.num_output_tokens = min(self.num_output_tokens + 1, self.max_candidates)
        elif candidate_num > num_matches:
            ran = random.random() + 0.1 * (candidate_num - num_matches)
            if ran > accept_rate:
                self.num_output_tokens = max(self.num_output_tokens - 1, self.min_candidates)


@torch.no_grad()
def lookup_generate(self,
                    inputs: Optional[torch.Tensor] = None,
                    max_new_tokens: int = 10,
                    num_output_tokens: int = 10,
                    max_matching_ngram_size: int = None,
                    generation_config: Optional[GenerationConfig] = None,
                    streamer: Optional["BaseStreamer"] = None,
                    attention_mask=None,
                    **sampling_kwargs):
    from packaging import version
    trans_version = transformers.__version__

    if version.parse(trans_version) >= version.parse("4.45.0"):
        input_ids, generation_config, logits_processor, stopping_criteria, \
            model_kwargs = _prepare_generate_args_4_45(self, inputs, generation_config,
                                                       streamer, **sampling_kwargs)
    else:
        input_ids, generation_config, logits_processor, stopping_criteria, \
            model_kwargs = _prepare_generate_args(self, inputs, generation_config,
                                                  streamer, **sampling_kwargs)

    invalidInputError(input_ids.shape[0] == 1,
                      "Prompt lookup is currently not supported with batch inference.")

    device_name = get_xpu_device_type(input_ids)

    candidates_generator = PromptLookupCandidateGenerator(
        num_output_tokens=num_output_tokens,
        max_matching_ngram_size=max_matching_ngram_size,
        device=device_name)

    step = 0
    step_verify = 0

    clear_benchmarks(self)
    self.accept_rate = []

    past_key_values = None
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

        if step == 0:
            # first token use full model
            tic = time.time()
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            output = self(**model_inputs,
                          return_dict=True)
            logits = output['logits']
            logits = logits[:, -1:]
            logits[:, -1, :] = logits_processor(input_ids, logits[:, -1, :])
            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
            else:
                output_ids = greedy(logits)
            input_ids = torch.cat((input_ids, output_ids), dim=-1)

            candidates_generator.init_look_up_table(input_ids)

            past_key_values = output['past_key_values']
            step += 1
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.first_token_time = toc - tic
            e2e_tic = time.time()
        else:
            cur_len = input_ids.shape[-1]
            toc = time.time()
            candidate_input_ids, _ = candidates_generator.get_candidates(input_ids=input_ids)
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            verify_input_ids = candidate_input_ids[:, -candidate_length - 1:]
            self.draft_num.append(candidate_length)
            tic = time.time()
            self.draft_time.append(tic - toc)
            if attention_mask is None:
                cur_attention_mask = None
            else:
                appended_len = verify_input_ids.size(1) + step - 1
                ones_to_append = torch.ones(attention_mask.size(0), appended_len,
                                            device=self.device)
                cur_attention_mask = torch.cat((attention_mask, ones_to_append), dim=1)
            output = _non_cpu_ipex_verify(self, verify_input_ids, past_key_values,
                                          cur_attention_mask, return_dict=True, use_cache=True)
            if isinstance(output, dict):
                logits = output['logits']
                past_key_values = output['past_key_values']

            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i],
                                                       logits[:, i, :])

            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
                output_ids = output_ids.transpose(0, 1)
            else:
                output_ids = greedy(logits)

            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.verify_time.append(toc - tic)

            # Compare drafts with target verified outputs
            # Drafts start from [1, k]
            # Verified output start from [0, k - 1]
            # including the one generated by the base model

            n_matches = ((output_ids[:, :-1] != verify_input_ids[:, 1:])
                         .cumsum(-1) == 0).sum(-1).item()

            max_matched = n_matches + 1
            mot = time.time()
            self.match_time.append(mot-toc)

            max_of_max_matched = output_ids.size(1)
            # Accept number is max_matched, min is 1
            self.accept_num.append(max_matched)
            self.n_matched += n_matches
            self.n_drafted += candidate_length

            # Clean up target model KV cache
            if max_of_max_matched != max_matched:
                output_ids = output_ids[:, :max_matched]
                new_cache_size = max_of_max_matched - max_matched
                past_key_values = _crop_past_key_values(self, past_key_values,
                                                        new_cache_size)

            accept_rate = self.n_matched/self.n_drafted if self.n_drafted > 0 else 1
            self.accept_rate.append(accept_rate)
            # Update the candidate generation strategy if needed
            if device_name not in ["mtl", "lnl"]:
                candidates_generator.update_candidate_strategy(candidate_length, n_matches,
                                                               accept_rate)

            input_ids = torch.cat((input_ids, output_ids), dim=-1)
            candidates_generator.update_look_up_table(input_ids)

            step += output_ids.size(1)
            step_verify += 1
            pot = time.time()
            self.post_time.append(pot-mot)

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
    e2e_toc = time.time()
    self.n_token_generated = step
    self.e2e_time_without_first = e2e_toc - e2e_tic

    if streamer is not None:
        streamer.end()

    return input_ids[:, : input_len + step]
