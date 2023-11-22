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
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/vllm/worker/worker.py
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
"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed
import warnings
import numpy as np
import random

from bigdl.llm.vllm.config import ModelConfig, SchedulerConfig
from bigdl.llm.vllm.model_executor.model_loader import get_model
from bigdl.llm.vllm.model_executor.input_metadata import InputMetadata
from bigdl.llm.vllm.sampling_params import SamplingParams
from bigdl.llm.vllm.sequence import SequenceData, SamplerOutput, SequenceGroupMetadata
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.vllm.model_executor.utils import set_random_seed


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

# bigdl-llm Intel specified code change
# bigdl-llm change start
# summary: Remove config for parallel and cache engine.
# Add kv_cache dict and methods to maintain.
    def __init__(
        self,
        model_config: ModelConfig,
        # parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        # distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        # self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        # self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        self.kv_cache = dict()

    def clean_finished_seqs(self, finished_seqs: List[int]):
        """
        This function cleans the finished sequences and their KVCache in self.kv_cache
        """
        for seq_id in finished_seqs:
            if seq_id not in self.kv_cache.keys():
                warnings.warn(f"Duplicate key {seq_id} received during clean worker's KVCache")
                continue
            del self.kv_cache[seq_id]

    def init_model(self):
        if self.model_config.device == 'gpu':
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            # Env vars will be set by Ray.
            self.rank = self.rank if self.rank is not None else int(
                os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                invalidInputError(False, "Invalid or unspecified rank.")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment.
            # Co(gc): Consider this later
            # _init_distributed_environment(self.parallel_config, self.rank,
            #                               self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            # if seq_group_metadata.block_tables is None:
            #     # During memory profiling, the block tables are not initialized
            #     # yet. In this case, we just use a dummy slot mapping.
            #     slot_mapping.extend([0] * prompt_len)
            #     continue

            # # Compute the slot mapping.
            # block_table = seq_group_metadata.block_tables[seq_id]
            # for i in range(prompt_len):
            #     block_number = block_table[i // self.block_size]
            #     block_offset = i % self.block_size
            #     slot = block_number * self.block_size + block_offset
            #     slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        # generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append(position)

                # block_table = seq_group_metadata.block_tables[seq_id]

                max_context_len = max(max_context_len, context_len)
                # max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                #                              len(block_table))
                context_lens.append(context_len)

                # block_number = block_table[position // self.block_size]
                # block_offset = position % self.block_size
                # slot = block_number * self.block_size + block_offset
                # slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                # generation_block_tables.append(block_table)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens,
                                     dtype=torch.long,
                                     # device="cuda"
                                     )
        positions_tensor = torch.tensor(input_positions,
                                        dtype=torch.long,
                                        # device="cuda"
                                        )
        # slot_mapping_tensor = torch.tensor(slot_mapping,
        #                                    dtype=torch.int,
        #                                    device="cuda")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           # device="cuda"
                                           )
        # padded_block_tables = [
        #     _pad_to_max(block_table, max_num_blocks_per_seq)
        #     for block_table in generation_block_tables
        # ]
        # block_tables_tensor = torch.tensor(padded_block_tables,
        #                                    dtype=torch.int,
        #                                    device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            # slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            # block_tables=block_tables_tensor,
            sliding_window=self.sliding_window,
        )
        return tokens_tensor, positions_tensor, input_metadata

    # TODO(gc): we may want to delete unused parameters
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        finished_seqs: List[int],
    ) -> SamplerOutput:
        # Issue cache operations.
        # issued_cache_op = False
        # if blocks_to_swap_in:
        #     self.cache_engine.swap_in(blocks_to_swap_in)
        #     issued_cache_op = True
        # if blocks_to_swap_out:
        #     self.cache_engine.swap_out(blocks_to_swap_out)
        #     issued_cache_op = True
        # if blocks_to_copy:
        #     self.cache_engine.copy(blocks_to_copy)
        #     issued_cache_op = True

        # if issued_cache_op:
        #     cache_events = self.cache_events
        # else:
        #     cache_events = None
        if finished_seqs:
            self.clean_finished_seqs(finished_seqs)

        cache_events = None
        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # pdb.set_trace()
        # TODO(gc): use environment/global virable to check
        if True:
            input_tokens, input_positions, input_metadata = self._prepare_inputs(
                seq_group_metadata_list)
            output = self.model(
                seq_group_meta_data_lists=seq_group_metadata_list,
                kv_cache=self.kv_cache, input_metadata=input_metadata)
            return output
        else:
            # Prepare input tensors.
            input_tokens, input_positions, input_metadata = self._prepare_inputs(
                seq_group_metadata_list)

            # Execute the model.
            output = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=None,
                input_metadata=input_metadata,
                cache_events=cache_events,
            )
        return output

# bigdl-llm change end


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            invalidInputError(
                False,
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
