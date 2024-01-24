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

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from bigdl.llm.transformers import AutoModelForCausalLM

class PPL:
    def __init__(self):
        self.nll = 0
        self.cnt = 0
    def __call__(self, all_logits, labels):
        '''
            all_logits [seq_length, vocab_size]
            labels [seq_length]
        '''
        seq_length = all_logits.shape[0]
        for i in range(0, seq_length - 1):
            logits = all_logits[i, :]
            max_logit = np.amax(logits)
            sum_exp = np.sum(np.exp(logits - max_logit))
            # logits at time-step i is for predicting token at time-step (i+1)
            next_tok = labels[i + 1]
            log_softmax_of_tok = (logits[next_tok] - max_logit) - np.log(sum_exp)
            self.nll += -log_softmax_of_tok
            self.cnt += 1
        return np.exp(self.nll / self.cnt)
   
    def result(self):
        return np.exp(self.nll / self.cnt)

    def __str__(self):
        return f"PPL: {np.exp(self.nll / self.cnt):.3f}"


class BigDLPPL:
    def __init__(self, model_path, device, **model_kwargs) -> None:
        model_kwargs['trust_remote_code'] = model_kwargs.get('trust_remote_code', True)
        model_kwargs['optimize_model'] = model_kwargs.get('optimize_model', True)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if 'xpu' in device:
            import intel_extension_for_pytorch as ipex
        self.model.to(device)
        self.ppl_evaluator = PPL()

    def perplexity_hf(self, text):
        inputs = self.tokenizer('\n\n'.join(text), return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        #    attention_mask = inputs['attention_mask']
        progress_bar = tqdm(range(0, input_ids.shape[1], 512))
        
        for i0 in progress_bar:
            input_ids_chunks = input_ids[:, i0:(i0+512)]
            input_ids_chunks[:, 0] = 1
            with torch.no_grad():
                result = self.model.forward(input_ids_chunks, labels = input_ids_chunks, return_dict=True)
                #print(f"ppl = {torch.exp(result.loss)}")
                seq_len = result.logits.shape[1]
                data = result.logits
                data = data.to('cpu')
                input_ids_chunks = input_ids_chunks.to('cpu')
                self.ppl_evaluator(data.numpy()[0, seq_len//2:, :], input_ids_chunks.numpy()[0, seq_len//2:])
            progress_bar.set_description(f"{self.ppl_evaluator}")

        return self.ppl_evaluator.result()
