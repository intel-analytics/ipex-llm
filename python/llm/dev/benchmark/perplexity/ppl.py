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
# This file is adapted from
# https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py
#

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import gc

from ipex_llm.transformers import AutoModelForCausalLM, AutoModel

class BigDLPPL:
    def __init__(self, model_path, device, **model_kwargs) -> None:
        model_kwargs['trust_remote_code'] = model_kwargs.get('trust_remote_code', True)
        model_kwargs['optimize_model'] = model_kwargs.get('optimize_model', True)
        self.device = device

        if 'chatglm' in model_path.lower():
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.to(device)


    def perplexity_hf(self, encoded_texts):
        self.model.eval()
        loss_fct = CrossEntropyLoss(reduction="none")
        ppls = []

        try:
            pbar = tqdm(range(len(encoded_texts)))
            for bid in pbar:
                encoded_batch = encoded_texts[bid:bid+1]
                if type(encoded_batch) == dict:
                    attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
                    encoded_batch = encoded_batch['input_ids']
                elif type(encoded_batch) == list:
                    encoded_batch = encoded_batch[0]
                
                encoded_batch = encoded_batch.to(self.device)
                attn_mask = torch.ones_like(encoded_batch)

                out_logits = self.model(encoded_batch).logits

                labels = encoded_batch

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

                loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
                perplexity_batch = torch.exp2(
                    (loss_ * shift_attention_mask_batch).sum(1)
                    / shift_attention_mask_batch.sum(1)
                )
                ppls += perplexity_batch.tolist()

                pbar.set_description(f"[{bid:<4}/{len(encoded_texts)}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")
                
                del out_logits, encoded_batch, attn_mask, shift_logits, shift_labels, shift_attention_mask_batch, perplexity_batch

            ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])
        finally:
            if self.device == "xpu":
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            del self.model
            gc.collect()
        
        return ppl_mean