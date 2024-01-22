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

from typing import Optional
import torch
import time
import os
import copy
import logging
import warnings
import inspect
from transformers import top_k_top_p_filtering, GenerationConfig, LogitsProcessorList, StoppingCriteriaList


def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids


def clear_benchmarks(self):
    self.generate_time = []
    self.draft_time = []
    self.verify_time = []
    self.draft_num = []
    self.accept_num = []
    self.n_drafted = 0
    self.n_matched = 0


@torch.no_grad()
def speculative_generate(self,
                         input_ids: Optional[torch.Tensor] = None,
                         draft_model = None,
                         max_new_tokens=10,
                         max_step_draft=8,
                         th_stop_draft=0.8,
                         auto_th_stop_draft=True,
                         auto_parameters=[1,0.5,0.9,1e-2,0.9],
                         do_sample=False,
                         top_k=0,
                         top_p=0.85,
                         temperature=0.2,
                         hf_adjust=False):
    assert draft_model is not None, "Draft model should be provided."
    step = 0
    step_draft = 0
    step_verify = 0

    draft_gen_length = max_step_draft+6 if hf_adjust else max_step_draft+1
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=self.device)
    draft_generate_ids = torch.empty([input_ids.size(0), draft_gen_length], dtype=torch.long, device=self.device)
    past_key_values = None

    tmp_matchness = 0
    e2e_tic = 0.0

    self.clear_benchmarks()

    if self.config.model_type == "qwen":
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
        logit_processor = RepetitionPenaltyLogitsProcessor(penalty=self.generation_config.repetition_penalty)
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
            output = self(input_ids=current_input_ids,
                            past_key_values=past_key_values,
                            return_dict=True,
                            use_cache=True)
            logits = output['logits']
            logits = logits[:,-1:]
            if self.config.model_type == "qwen":
                temp_input_ids = torch.cat((input_ids, generate_ids[:, :step]), dim=-1)
                logits[:, -1, :] = logit_processor(temp_input_ids, logits[:, -1, :])
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output['past_key_values']
            step += 1
            e2e_tic = time.time()
        else:
            draft_current_input_ids = current_input_ids
            # Target model KV cache to draft model
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
                if self.config.model_type == "qwen":
                    temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                                draft_generate_ids[:, 1:step_draft+1]), dim=-1)
                    draft_output['logits'][ :, -1, : ] = logit_processor(temp_input_ids,
                                                                         draft_output['logits'][:, -1, :])
                draft_output_ids, draft_output_probs = sample(
                    draft_output['logits'], return_probs=True, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
                draft_generate_ids[:, step_draft+1] = draft_output_ids
                draft_current_input_ids = draft_output_ids
                draft_past_key_values = draft_output['past_key_values']
                # check if draft prob is less then th_stop_draft
                # Draft number + step >= max output token number
                if draft_output_probs.item() < th_stop_draft or step + step_draft + 2 >= max_new_tokens:
                    break
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.draft_time.append(toc - tic)
            if self.do_print:
                print(f"Step {step} Draft time {self.draft_time[-1]}")
            drafted_n_tokens = step_draft + 1
            drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1] # raft input + raft completion
            self.draft_num.append(drafted_n_tokens)
            tic = time.time()
            # Target model verify drafts
            # input.size is k + 1, 1 previous token + k drafts
            # verified output.size is k + 1, k token + 1 final
            # Final token is always accepted
            if self.config.model_type == "chatglm":
                past_key_value_len = past_key_values[0][0].shape[0]
                position_ids = torch.arange(drafted_input_ids.shape[1], dtype=torch.long,
                                            device=drafted_input_ids.device).unsqueeze(0).repeat(1, 1) + past_key_value_len
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
            if self.config.model_type == "qwen":
                temp_input_ids = torch.cat((input_ids, generate_ids[:, :step],
                                            draft_generate_ids[:, 1:step_draft + 2]), dim=-1)
                for i in range(logits.size(1)):
                    logits[:, i, :] = logit_processor(temp_input_ids
                        [:, : input_ids.size(1) + step + i], output['logits'][:, i, :])
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            toc = time.time()
            self.verify_time.append(toc - tic)
            if self.do_print:
                print(f"Step {step} Verify time {self.verify_time[-1]}")
            self.generate_time.append(self.draft_time[-1] + self.verify_time[-1])
            if self.do_print:
                print(f"Step {step} Generation time {self.generate_time[-1]}")

            past_key_values = output['past_key_values']
            # Compare drafts with target verified outputs
            # Drafts start from [1, k]
            # Verified output start from [0, k - 1]
            # including the one generated by the base model
            max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
            max_of_max_matched = output_ids.size(1)
            # Accept number is max_matched, min is 1
            self.accept_num.append(max_matched)
            if max_matched != 1:
                if self.do_print:
                    print(f"Step {step} Matched {max_matched}")
            else:
                if self.do_print:
                    print(f"Step {step} Rejected")
            # Clean up target model KV cache
            if max_of_max_matched != max_matched:
                output_ids = output_ids[:, :max_matched]
                # For Qwen
                if self.config.model_type == "qwen":
                    past_key_values = [
                        (k[:, :-(max_of_max_matched - max_matched), :], v[:, :-(max_of_max_matched - max_matched), :]) for k, v in past_key_values
                    ]
                elif self.config.model_type == "chatglm":
                    # for chatglm, cache shape is [sl, bs, nh, hn]
                    past_key_values = [
                        (k[ :-(max_of_max_matched - max_matched), :, :, :], v[ :-(max_of_max_matched - max_matched), :, :, :]) for k, v in past_key_values
                    ]
                elif self.config.model_type == "baichuan":
                    past_key_values = [
                        (k[ :, :, :-(max_of_max_matched - max_matched), :], v[ :, :, :-(max_of_max_matched - max_matched), :]) for k, v in past_key_values
                    ]
                else:
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
                    ]

            generate_ids[:, step:step+output_ids.size(1)] = output_ids
            current_input_ids = output_ids[:, -1:]

            step += output_ids.size(1)

            # remove one generated by the base model
            self.n_matched += max_matched - 1
            self.n_drafted += drafted_n_tokens
            step_verify += 1

            if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                tmp_matchness = auto_parameters[1]*(tmp_matchness) + (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                if tmp_matchness<auto_parameters[2]:
                    new_th_stop_draft = th_stop_draft+auto_parameters[3]
                else:
                    if drafted_n_tokens==max_step_draft:
                        new_th_stop_draft = th_stop_draft
                    else:
                        new_th_stop_draft = th_stop_draft-auto_parameters[3]
                th_stop_draft = auto_parameters[4] * th_stop_draft + (1-auto_parameters[4]) * new_th_stop_draft
                # print('draft_output_probs: {:.4f}, th_stop_draft: {:.4f}, tmp_matchness: {:.2f}, drafted_n_tokens: {:d}'.format(
                #     draft_output_probs.item(), th_stop_draft, tmp_matchness, drafted_n_tokens))
            
            if hf_adjust:
                if (max_matched - 1) == max_step_draft:
                    max_step_draft = min(draft_gen_length - 1, max_step_draft + 1)
                else:
                    max_step_draft = max(1, max_step_draft - 1)

        # Stop on eos and remove content after eos
        output_ids_list = output_ids[0].tolist()
        if self.config.eos_token_id in output_ids_list:
            idx = output_ids_list.index(self.config.eos_token_id)
            step -= (len(output_ids_list) - idx - 1)
            break

    step = min(step, max_new_tokens)
    e2e_toc = time.time()
    self.n_token_generated = step
    self.e2e_time_without_first = e2e_toc - e2e_tic

    if self.do_print:
        print(f"Final token number {self.n_token_generated}")
        print(f"Average Draft time {sum(self.draft_time)/self.n_drafted}")
        print(f"Average Verify time {sum(self.verify_time)/len(self.verify_time)}")
        print(f"Average Generation time {sum(self.generate_time)/len(self.generate_time)}")
        print(f"Generation throughput {1.0 * (step - 1) / sum(self.generate_time)}")
        print(f"E2E Generation throughput without first token {1.0 * (step - 1) / self.e2e_time_without_first }")
        print(f"Draft num {self.n_drafted}")
        print(f"Accept num {self.n_matched}")
        print(f"Draft len: {self.n_drafted/len(self.draft_num)}, accept len: {self.n_matched/len(self.accept_num)}")
        print(f"Draft {self.draft_num}")
        print(f"Accept {self.accept_num}")
        print(f"Generation time {self.generate_time}")

    generate_ids = torch.cat([input_ids, generate_ids[:, :step]], dim=-1)

    return generate_ids