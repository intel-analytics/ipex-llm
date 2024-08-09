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


import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


# todo
def patched_repetition_penalty_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    score = torch.gather(scores, 1, input_ids)

    # if score < 0 then repetition penalty has to be
    # multiplied to reduce the token probabilities
    score = torch.where(score < 0, score * self.penalty, score / self.penalty)

    # ipex llm changes start: call scatter on CPU
    device = scores.device
    scores = scores.to('cpu')
    input_ids = input_ids.to('cpu')
    score = score.to('cpu')
    scores.scatter_(1, input_ids, score)
    scores = scores.to(device)
    # ipex llm changes end

    return scores


def minicpmv_generate_wrapper(origin_generate):
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        RepetitionPenaltyLogitsProcessor.__call__ = patched_repetition_penalty_call
        return origin_generate(
            self=self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            tgt_sizes=tgt_sizes,
            image_bound=image_bound,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            vision_hidden_states=vision_hidden_states,
            return_vision_hidden_states=return_vision_hidden_states,
            stream=stream,
            decode_text=decode_text,
            **kwargs
        )
    return generate
