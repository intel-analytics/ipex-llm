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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
# which is licensed under Apache License 2.0
#
# Copyright 2023 The vLLM team. All rights reserved.
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

import torch
from torch import nn

from transformers import AutoTokenizer, PreTrainedTokenizerBase, LlamaConfig
from typing import Optional, Tuple, List, Type, Dict

from bigdl.llm.vllm.structure.sequence import SequenceOutputs, SequenceGroupMetadata
from .bigdl_sampler import BigDLSampler
import math
import time

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def prepare_logits_processor(temperature: float, repetition_penalty: float,
                             top_p: float, top_k: int) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    # if repetition_penalty > 1.0:
    #     processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def _pad_to_max(x: List[int], max_len: int, padding_id: int = 0) -> List[int]:
    return x + [padding_id] * (max_len - len(x))


def _pad_kv_cache_view(t: torch.Tensor, len: int,
                       device: torch.device, pos: int = 2) -> torch.Tensor:
    cur_size = list(t.size())
    if cur_size[pos] < len:
        tmp_size = cur_size[:]
        tmp_size[pos] = len - cur_size[pos]
        zeros = torch.zeros(tmp_size, device=device)
        padded_view = torch.cat((zeros, t), dim=pos)
        return padded_view
    if cur_size[pos] > len:
        padded_view = t.narrow(pos, cur_size[pos] - len, len)
        return padded_view
    return t


class BigDLLlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        device: Optional[str] = None,
    ):
        super().__init__()
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
                print("Intel Extension for PyTorch is not installed, but is required for xpu inference.")

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
            self.model.to(device)
            self.device = torch.device(device)
        self.dtype = self.model.dtype
        self.kv_cache_size = [0]
        self.last_seq_ids = []
        self.tmp_kv_cache = None
        self.pad_token_id = config.pad_token_id
        self.max_seq_limit = 128

    def forward(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Optional = None,
        input_metadata: Optional = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        kv_cache_0 = self.model.config.num_hidden_layers
        kv_cache_1 = 2
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

            context_len = seq_data.get_len()

        if all_decoding:
            # pdb.set_trace()
            max_seq_limit = self.max_seq_limit
            if (self.tmp_kv_cache is not None) and cur_seq_ids == self.last_seq_ids:
                if self.tmp_kv_cache[0][0].size(2) < max_seq_limit:
                    bigdl_kv_cache = self.tmp_kv_cache
                else:
                    bigdl_kv_cache = [[tmp.narrow(2, self.tmp_kv_cache[0][0].size(2)
                                       - max_seq_limit, max_seq_limit)
                                       for tmp in tmp_list] for tmp_list in self.tmp_kv_cache]
            else:
                bigdl_kv_cache = []
                for i in range(kv_cache_0):
                    cur_list = []
                    for j in range(kv_cache_1):
                        cur_view = None
                        for seq_group_meta_data in seq_group_meta_data_lists:
                            seq_ids = list(seq_group_meta_data.seq_data.keys())
                            seq_id = seq_ids[0]
                            seq_data = seq_group_meta_data.seq_data[seq_id]
                            view_size = [1] + list(kv_cache[seq_id][i][j].shape)
                            if cur_view is None:
                                cur_view = kv_cache[seq_id][i][j].view(view_size)
                            else:
                                if cur_view.size(2) != view_size[2]:
                                    max_len = max(cur_view.size(2), view_size[2])
                                    cur_view = _pad_kv_cache_view(cur_view, max_len, self.device)
                                    tmp_view = _pad_kv_cache_view(
                                        kv_cache[seq_id][i][j].view(view_size),
                                        max_len, self.device)
                                    cur_view = torch.cat((cur_view, tmp_view), dim=0)
                                else:
                                    cur_view = torch.cat(
                                        (cur_view, kv_cache[seq_id][i][j].view(view_size)), dim=0)
                        if cur_view.size(2) > max_seq_limit:
                            cur_view = _pad_kv_cache_view(cur_view, max_seq_limit, self.device)
                        cur_list.append(cur_view)
                    bigdl_kv_cache.append(cur_list)
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

        self.tmp_kv_cache = outputs.past_key_values
        self.kv_cache_size = list(outputs.past_key_values[0][0].shape)
        logits = outputs.logits[:, -1, :]
        bigdl_output = self.sampler(logits, input_metadata, st_timestamp)

        index = 0
        for seq_id in cur_seq_ids:
            if kv_cache.get(seq_id) is None:
                kv_cache[seq_id] = [[[] for _ in range(kv_cache_1)]
                                    for _ in range(kv_cache_0)]
            for i in range(kv_cache_0):
                for j in range(kv_cache_1):
                    kv_cache[seq_id][i][j] = outputs.past_key_values[i][j][
                        index]
            index = index + 1

        return bigdl_output

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        pass
