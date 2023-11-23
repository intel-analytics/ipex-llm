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
from torch import nn

from transformers import AutoTokenizer, PreTrainedTokenizerBase, LlamaConfig
from typing import Optional, Tuple, List, Type, Dict

from bigdl.llm.vllm.sequence import SequenceOutputs, SequenceGroupMetadata
from bigdl.llm.vllm.model_executor.layers.bigdl_sampler import BigDLSampler
from bigdl.llm.vllm.model_executor.models.bigdl_model import BigDLModelForCausalLM
import math
import time

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def _pad_to_max(x: List[int], max_len: int, padding_id: int = 0) -> List[int]:
    return x + [padding_id] * (max_len - len(x))


class BigDLLlamaForCausalLM(BigDLModelForCausalLM):

    def __init__(
        self,
        config: LlamaConfig,
        device: Optional[str] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__(config, device, max_model_len)
        self.config = config
        # TODO(gc): later change this to a switch?
        if True:
            from bigdl.llm.transformers import AutoModelForCausalLM
            from bigdl.llm import optimize_model

        # low_bit = 'sym_int4'
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=True,
            )
            self.model = optimize_model(model)
            self.sampler = BigDLSampler(config.vocab_size, device)
        elif device == 'xpu':
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                print("Intel Extension for PyTorch is not installed, \
                       but is required for xpu inference.")

            low_bit = 'sym_int4'
            model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                load_in_low_bit=low_bit,
                trust_remote_code=True,
                use_cache=True,
            )
            self.model = model.to('xpu')
            self.sampler = BigDLSampler(config.vocab_size, device).to('xpu')

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = self.model.dtype
        self.kv_cache_size = [0]
        self.last_seq_ids = []
        self.tmp_kv_cache = None
        self.pad_token_id = config.pad_token_id
        self.max_seq_limit = max_model_len

    def forward(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Optional = None,
        input_metadata: Optional = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        kv_cache_size_0 = self.model.config.num_hidden_layers
        kv_cache_size_1 = 2
        seq_len = len(seq_group_meta_data_lists)

        bigdl_input_ids = []
        bigdl_position_ids = []
        bigdl_attention_mask = []

        cur_seq_ids = []
        bigdl_sampling_params = {}
        max_context_len = 0
        all_decoding = True
        for seq_group_meta_data in seq_group_meta_data_lists:
            req_id = seq_group_meta_data.request_id
            all_decoding = all_decoding and (not seq_group_meta_data.is_prompt)
            seq_ids = list(seq_group_meta_data.seq_data.keys())
            seq_id = seq_ids[0]
            cur_seq_ids.append(seq_id)
            seq_data = seq_group_meta_data.seq_data[seq_id]

            cur_seq_input_ids = seq_data.get_token_ids()
            context_len = seq_data.get_len()
            if seq_group_meta_data.is_prompt:
                bigdl_input_ids.append(cur_seq_input_ids)
                max_context_len = max(max_context_len, context_len)
            else:
                bigdl_input_ids.append([cur_seq_input_ids[-1]])

            bigdl_sampling_params[seq_id] = seq_group_meta_data.sampling_params

        if all_decoding:
            bigdl_kv_cache = self.prepare_kv_cache(cur_seq_ids, seq_group_meta_data_lists,
                                                   kv_cache, kv_cache_size_0, kv_cache_size_1)
        else:
            bigdl_input_ids = [
                _pad_to_max(input_ids, max_context_len, self.pad_token_id)
                for input_ids in bigdl_input_ids
            ]

        if all_decoding:
            cur_seq_len = bigdl_kv_cache[0][0].size(2)
            for seq_group_meta_data in seq_group_meta_data_lists:
                seq_ids = list(seq_group_meta_data.seq_data.keys())
                seq_id = seq_ids[0]
                seq_data = seq_group_meta_data.seq_data[seq_id]
                cur_pos = seq_data.get_len()
                bigdl_position_ids.append([cur_pos - 1])
                cur_attention_mask = [0] * (cur_seq_len - cur_pos + 1) + [1] * (cur_pos)
                bigdl_attention_mask.append(cur_attention_mask)

        bigdl_input_ids = torch.tensor(bigdl_input_ids, device=self.device)
        if all_decoding:
            bigdl_position_ids = torch.tensor(bigdl_position_ids, device=self.device)
            bigdl_attention_mask = torch.tensor(bigdl_attention_mask, device=self.device)
            kwargs = {
                "input_ids": bigdl_input_ids,
                "position_ids": bigdl_position_ids,
                "attention_mask": bigdl_attention_mask,
                "past_key_values": bigdl_kv_cache,
                "use_cache": True,
                # "return_dict": True,
            }
        else:
            kwargs = {
                "input_ids": bigdl_input_ids,
                # "position_ids": bigdl_position_ids,
                "past_key_values": None,
                "use_cache": True,
                # "return_dict": True,
            }
        # pdb.set_trace()
        st_timestamp = time.perf_counter()
        outputs = self.model.forward(**kwargs)

        self.last_seq_ids = cur_seq_ids[:]
        self.tmp_kv_cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        bigdl_output = self.sampler(logits, input_metadata, st_timestamp)

        self.update_kv_cache(cur_seq_ids, outputs.past_key_values,
                             kv_cache, kv_cache_size_0, kv_cache_size_1)

        return bigdl_output
