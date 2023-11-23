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
from typing import Optional, Tuple, List, Type, Dict
from transformers import LlamaConfig

from bigdl.llm.vllm.sequence import SequenceOutputs, SequenceGroupMetadata


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


class BigDLModelForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        device: Optional[str] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.max_seq_limit = max_model_len
        self.last_kv_cache = None
        self.last_seq_ids = None

    # This is an implementation for models that KV Cache shape in (batch_size, num_heads,
    # sequence_length, embed_size_per_head).
    def prepare_kv_cache(
        self,
        cur_seq_ids: List[int],
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Dict,
        kv_cache_size_0: int,
        kv_cache_size_1: int,
    ):
        max_seq_limit = self.max_seq_limit
        if (self.last_kv_cache is not None) and cur_seq_ids == self.last_seq_ids:
            if self.last_kv_cache[0][0].size(2) < max_seq_limit:
                bigdl_kv_cache = self.tmp_kv_cache
            else:
                bigdl_kv_cache = [[tmp.narrow(2, self.last_kv_cache[0][0].size(2)
                                   - max_seq_limit, max_seq_limit)
                                   for tmp in tmp_list] for tmp_list in self.last_kv_cache]
        else:
            bigdl_kv_cache = []
            for i in range(kv_cache_size_0):
                cur_list = []
                for j in range(kv_cache_size_1):
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
        return bigdl_kv_cache

    # This is an implementation for models that KV Cache shape in (batch_size, num_heads,
    # sequence_length, embed_size_per_head).
    def update_kv_cache(
        self,
        cur_seq_ids: List[int],
        past_key_values: List[List[torch.Tensor]],
        kv_cache: Dict,
        kv_cache_size_0: int,
        kv_cache_size_1: int,
    ) -> None:
        index = 0
        for seq_id in cur_seq_ids:
            if kv_cache.get(seq_id) is None:
                kv_cache[seq_id] = [[[] for _ in range(kv_cache_size_1)]
                                    for _ in range(kv_cache_size_0)]
            for i in range(kv_cache_size_0):
                for j in range(kv_cache_size_1):
                    kv_cache[seq_id][i][j] = past_key_values[i][j][index]
            index = index + 1

    def forward(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Optional = None,
        input_metadata: Optional = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        pass

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        pass
