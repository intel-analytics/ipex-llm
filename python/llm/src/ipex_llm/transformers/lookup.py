from typing import Optional, Tuple
import torch
import time
import copy
import logging
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ipex_llm.transformers.speculative import greedy, deepmind_sample, logits_to_probs
from ipex_llm.utils.common import invalidInputError

logger = logging.getLogger("ipex_llm.lookup")


class PromptLookupCandidateGenerator():
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

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
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.
        """
        input_length = input_ids.size(1)

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Currently does nothing
        return
    

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
def lookup_generate(self,
                    inputs: Optional[torch.Tensor] = None,
                    max_new_tokens: int = 10,
                    num_output_tokens: int = 10,
                    max_matching_ngram_size: int = None,
                    tokenizer=None,
                    generation_config: Optional[GenerationConfig] = None,
                    attention_mask=None,
                    **sampling_kwargs):

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

    candidates_generator = PromptLookupCandidateGenerator(
        num_output_tokens=num_output_tokens,
        max_matching_ngram_size=max_matching_ngram_size)

    step = 0
    step_verify = 0

    clear_benchmarks(self)

    current_input_ids = input_ids
    # generate_ids = torch.empty([input_ids.size(0), max_new_tokens+num_output_tokens],
    #                            dtype=torch.long, device=self.device)
    
    past_key_values = None
    input_len = input_ids.shape[1]

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
            # logits[:, -1, :] = logits_processor(current_input_ids, logits[:, -1, :])
            if generation_config.do_sample:
                output_ids, prob_list = deepmind_sample(logits,
                                                        top_k=generation_config.top_k,
                                                        top_p=generation_config.top_p,
                                                        temperature=generation_config.temperature)
            else:
                output_ids = greedy(logits)
            # generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            input_ids = torch.cat((input_ids, current_input_ids), dim=-1)
            past_key_values = output['past_key_values']
            step += 1
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.first_token_time = toc - tic
            e2e_tic = time.time()
        else:
            cur_len = input_ids.shape[-1]
            candidate_input_ids, _ = candidates_generator.get_candidates(input_ids=input_ids)
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            verify_input_ids = candidate_input_ids[:, -candidate_length - 1: ]
            # if candidate_length > 0:
            #     candidate_str = tokenizer.batch_decode(candidate_input_ids[:, -candidate_length - 2: ], skip_special_tokens=True)
            #     print(f"candidate_str: {candidate_str}")
            self.draft_num.append(candidate_length)
            forward_args = {
                "input_ids": verify_input_ids,
                "past_key_values": past_key_values,
                # "attention_mask": cur_attention_mask,
                "return_dict": True,
                "use_cache": True,
                }
            tic = time.time()
            output = self(**forward_args)
            if isinstance(output, dict):
                logits = output['logits']
                past_key_values = output['past_key_values']
        
            new_logits = logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            if generation_config.do_sample:
                target_probs = logits_to_probs(new_logits,
                                               top_k=generation_config.top_k,
                                               top_p=generation_config.top_p,
                                               temperature=generation_config.temperature)
            else:
                output_ids = greedy(new_logits)

            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.verify_time.append(toc - tic)
            
            # self.generate_time.append(self.draft_time[-1] + self.verify_time[-1])

            # Compare drafts with target verified outputs
            # Drafts start from [1, k]
            # Verified output start from [0, k - 1]
            # including the one generated by the base model
            max_matched = ((output_ids[:, :-1] != verify_input_ids[:, 1:]).cumsum(-1) == 0)
            max_matched = max_matched.sum(-1).item() + 1
            
            max_of_max_matched = output_ids.size(1)
            # Accept number is max_matched, min is 1
            self.accept_num.append(max_matched)
            self.n_matched += max_matched - 1
            self.n_drafted += candidate_length
            
            # print(f"verify time: {(toc - tic) * 1000} ms, candidate_length: {candidate_length}, max_matched: {max_matched}")

            # Clean up target model KV cache
            if max_of_max_matched != max_matched:
                output_ids = output_ids[:, :max_matched]

                past_key_values = [
                    (k[:, :, :-(max_of_max_matched - max_matched)],
                        v[:, :, :-(max_of_max_matched - max_matched)])
                    for k, v in past_key_values
                ]

            # generate_ids[:, step:step+output_ids.size(1)] = output_ids
            input_ids = torch.cat((input_ids, output_ids), dim=-1)

            step += output_ids.size(1)

            step_verify += 1

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

    # generate_ids = torch.cat([input_ids, generate_ids[:, :step]], dim=-1)

    return input_ids[:, : input_len + step]
