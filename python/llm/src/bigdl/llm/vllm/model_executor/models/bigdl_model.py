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
from bigdl.llm.transformers.models.utils import extend_kv_cache
from bigdl.llm.vllm.logger import init_logger

logger = init_logger(__name__)


zero_cache_dict = {}


def get_zero_tensor(length, cur_size, device, pos):
    if length not in zero_cache_dict:
        tmp_size = cur_size[:]
        tmp_size[pos] = length
        zero_cache_dict[length] = torch.zeros(tmp_size, device=device)
    return zero_cache_dict[length].narrow(pos, 0, length - cur_size[pos])


def _pad_kv_cache_view(t: torch.Tensor, len: int,
                       device: torch.device, pos: int = 2) -> torch.Tensor:
    cur_size = list(t.size())
    if cur_size[pos] < len:
        zeros = get_zero_tensor(len, cur_size, device, pos)
        padded_view = torch.cat((zeros, t), dim=pos)
        return padded_view
    elif cur_size[pos] > len:
        padded_view = t.narrow(pos, cur_size[pos] - len, len)
        return padded_view
    else:
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
            if device == 'xpu':
                try:
                    import intel_extension_for_pytorch as ipex
                except ImportError:
                    print("Intel Extension for PyTorch is not installed, \
                        but is required for xpu inference.")

        self.max_seq_limit = max_model_len
        self.last_kv_cache = None
        self.last_seq_ids = None

    def _set_last_kv_cache(self, last_kv_cache):
        self.last_kv_cache = last_kv_cache

    def _set_last_seq_ids(self, last_seq_ids):
        self.last_seq_ids = last_seq_ids

    # This is an implementation for models that KV Cache shape in (batch_size, num_heads,
    # sequence_length, embed_size_per_head).
    def prepare_kv_cache(
        self,
        cur_seq_ids: List[int],
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Dict,
        num_layers: int,
        kv_cache_size_1: int,
    ):
        max_seq_limit = self.max_seq_limit
        if (self.last_kv_cache is not None) and cur_seq_ids == self.last_seq_ids:
            if self.last_kv_cache[0][0].size(2) < max_seq_limit * 1.5:
                bigdl_kv_cache = self.last_kv_cache
                # Immediately set it to None to decrease ref-count
                self.last_kv_cache = None
            else:
                bigdl_kv_cache = [[tmp.narrow(2, self.last_kv_cache[0][0].size(2)
                                   - max_seq_limit, max_seq_limit)
                                   for tmp in tmp_list] for tmp_list in self.last_kv_cache]
                del self.last_kv_cache
                self.last_kv_cache = None
        else:
            del self.last_kv_cache
            bigdl_kv_cache = []
            max_kv_len = max(
                seq_group_meta_data.seq_data[next(iter(seq_group_meta_data.seq_data))].get_len()
                for seq_group_meta_data in seq_group_meta_data_lists
            )
            max_kv_len = min(max_kv_len, max_seq_limit)

            for i in range(num_layers):
                cur_list = []
                for j in range(kv_cache_size_1):
                    views = []
                    for seq_group_meta_data in seq_group_meta_data_lists:
                        seq_ids = list(seq_group_meta_data.seq_data.keys())
                        seq_id = seq_ids[0]
                        view_size = [1] + list(kv_cache[i][j][seq_id].shape)
                        views.append(kv_cache[i][j][seq_id].view(view_size))

                    views = [_pad_kv_cache_view(v, max_kv_len, self.device) for v in views]
                    cur_view = torch.cat(views, dim=0)
                    cur_list.append(cur_view)

                    for seq_group_meta_data in seq_group_meta_data_lists:
                        seq_ids = list(seq_group_meta_data.seq_data.keys())
                        seq_id = seq_ids[0]
                        del kv_cache[i][j][seq_id]

                bigdl_kv_cache.append(cur_list)

        return bigdl_kv_cache

    def get_construct_kv_cache_func(self, enable_selective_batching):
        if enable_selective_batching:
            return self.prepare_kv_cache_selective_batching
        else:
            return self.prepare_kv_cache

    # This is an implementation for models that KV Cache shape in (batch_size, num_heads,
    # sequence_length, embed_size_per_head).
    def prepare_kv_cache_selective_batching(
        self,
        cur_seq_ids: List[int],
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Dict,
        num_layers: int,
        kv_cache_size_1: int,
    ):
        # Return bigdl_kv_cache in the format of Tuple(List[Tuple(torch.Tensor)])
        bigdl_kv_cache = []
        for i in range(num_layers):
            # Construct a list of tuple(tensor)
            temp_cache = []
            for seq_id in cur_seq_ids:
                key = kv_cache[i][0][seq_id]
                value = kv_cache[i][1][seq_id]
                temp_cache.append((key, value))
            bigdl_kv_cache.append(temp_cache)
        return bigdl_kv_cache

    # This is an implementation for models that KV Cache shape in (batch_size, num_heads,
    # sequence_length, embed_size_per_head).
    def update_kv_cache(
        self,
        cur_seq_ids: List[int],
        kv_cache,
        layer: int,
        kv_cache_size_1: int,
    ) -> None:
        for i in range(layer):
            for j in range(kv_cache_size_1):
                batch_dim = 0
                for seq_id in cur_seq_ids:
                    kv_cache[i][j][seq_id] = self.last_kv_cache[i][j][batch_dim]
                    batch_dim = batch_dim + 1

    def update_kv_cache_selective_batching(
        self,
        cur_seq_ids: List[int],
        kv_cache,
        layer: int,
        kv_cache_size_1: int,
    ) -> None:
        for i in range(layer):
            for j in range(len(cur_seq_ids)):
                kv_cache[i][0][cur_seq_ids[j]] = self.last_kv_cache[i][j][0]
                kv_cache[i][1][cur_seq_ids[j]] = self.last_kv_cache[i][j][1]

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
