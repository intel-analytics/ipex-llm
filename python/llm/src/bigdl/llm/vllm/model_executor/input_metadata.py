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
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/vllm/model_executor/input_metadata.py
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

from typing import Dict, List, Optional, Tuple
import torch
from xformers.ops import AttentionBias
from bigdl.llm.vllm.sequence import SequenceData
from bigdl.llm.vllm.sampling_params import SamplingParams


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        context_lens: torch.Tensor,
        max_context_len: int,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.context_lens = context_lens
        self.max_context_len = max_context_len

        self.to_cache = None

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]

        # Set during the execution of the first attention op.
        # TODO(gc): we might want to delete this
        self.attn_bias: List[AttentionBias] = []

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}), '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}), '
                f'slot_mapping={self.slot_mapping}')
