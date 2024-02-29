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
from bigdl.llm.vllm.logger import init_logger
import math
import time
from bigdl.llm.vllm.model_executor.input_metadata import InputMetadata
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

logger = init_logger(__name__)


def _pad_to_max(x: List[int], max_len: int, padding_id: int = 0) -> List[int]:
    return [padding_id] * (max_len - len(x)) + x


def _get_attention_mask_for_prompts(
    input_ids: List[List[int]], max_prompt_len: int
) -> List[List[int]]:
    attention_mask = [
        [0] * (max_prompt_len - len(prompt)) + [1] * len(prompt) for prompt in input_ids
    ]
    return attention_mask


class BigDLMixtralForCausalLM(BigDLModelForCausalLM):

    def __init__(
        self,
        config,
        device: Optional[str] = None,
        max_model_len: Optional[int] = None,
        load_in_low_bit: str = 'sym_int4'
    ):
        super().__init__(config, device, max_model_len)
        self.config = config
        # TODO(gc): later change this to a switch?
        from bigdl.llm.transformers import AutoModelForCausalLM

        torch_dtype = 'auto'

        if load_in_low_bit == 'bf16':
            torch_dtype = torch.bfloat16
        elif load_in_low_bit == 'fp16':
            torch_dtype = torch.float16

        if device == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                load_in_low_bit=load_in_low_bit,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=True,
            )
            self.sampler = BigDLSampler(config.vocab_size, device)
        elif device == 'xpu':
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                print("Intel Extension for PyTorch is not installed, \
                       but is required for xpu inference.")

            model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                load_in_low_bit=load_in_low_bit,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                optimize_model=True,
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
        self.last_seq_ids = []
        self.tmp_kv_cache = None
        if config.pad_token_id is None:
            self.pad_token_id = config.eos_token_id
        else:
            self.pad_token_id = config.pad_token_id
        self.max_seq_limit = max_model_len

    def forward(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        # kv_cache in the format [[dict() for _ in range(2)] for _ in range(32)]
        kv_cache: Optional[List[List[Dict]]] = None,
        input_metadata: Optional[InputMetadata] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        num_layers = self.model.config.num_hidden_layers
        # One for key, one for value
        decoder_kv_size = 2

        bigdl_input_ids = []
        bigdl_position_ids = []
        bigdl_attention_mask = []

        cur_seq_ids = []
        max_prompt_len = 0

        # 0. Verify is_prompt or is_decoding
        is_decoding_stage = not seq_group_meta_data_lists[0].is_prompt

        # 1. Assemble bigdl_input_ids
        for seq_group_meta_data in seq_group_meta_data_lists:
            # req_id = seq_group_meta_data.request_id
            # is_decoding_stage = is_decoding_stage and (not seq_group_meta_data.is_prompt)
            seq_ids = list(seq_group_meta_data.seq_data.keys())
            seq_id = seq_ids[0]
            cur_seq_ids.append(seq_id)
            seq_data = seq_group_meta_data.seq_data[seq_id]

            cur_seq_input_ids = seq_data.get_token_ids()
            # context_len = seq_data.get_len()
            if seq_group_meta_data.is_prompt:
                bigdl_input_ids.append(cur_seq_input_ids)
                max_prompt_len = max(max_prompt_len, seq_data.get_len())
            else:
                bigdl_input_ids.append([cur_seq_input_ids[-1]])
        # 1. Assemble bigdl_input_ids end

        if is_decoding_stage:
            bigdl_kv_cache = self.prepare_kv_cache(cur_seq_ids, seq_group_meta_data_lists,
                                                   kv_cache, num_layers, decoder_kv_size)
        else:
            bigdl_attention_mask = _get_attention_mask_for_prompts(bigdl_input_ids, max_prompt_len)
            bigdl_input_ids = [
                _pad_to_max(input_ids, max_prompt_len, self.pad_token_id)
                for input_ids in bigdl_input_ids
            ]

        if is_decoding_stage:
            cur_seq_len = bigdl_kv_cache[0][0].size(2)
            for seq_group_meta_data in seq_group_meta_data_lists:
                seq_ids = list(seq_group_meta_data.seq_data.keys())
                seq_id = seq_ids[0]
                seq_data = seq_group_meta_data.seq_data[seq_id]
                cur_pos = seq_data.get_len()
                # bigdl_position_ids.append([cur_pos - 1])
                cur_attention_mask = [0] * (cur_seq_len - cur_pos + 1) + [1] * (cur_pos)
                bigdl_attention_mask.append(cur_attention_mask)

        bigdl_input_ids = torch.tensor(bigdl_input_ids, device=self.device)

        if is_decoding_stage:
            # bigdl_position_ids = torch.tensor(bigdl_position_ids, device=self.device)
            bigdl_attention_mask = torch.tensor(bigdl_attention_mask, device=self.device)
            kwargs = {
                "input_ids": bigdl_input_ids,
                # "position_ids": bigdl_position_ids,
                "attention_mask": bigdl_attention_mask,
                "past_key_values": bigdl_kv_cache,
                "use_cache": True,
                # "return_dict": True,
            }
        else:
            kwargs = {
                "input_ids": bigdl_input_ids,
                "attention_mask": torch.tensor(bigdl_attention_mask, device=self.device),
                # "position_ids": bigdl_position_ids,
                "past_key_values": None,
                "use_cache": True,
                # "return_dict": True,
            }
            if self.last_kv_cache:
                self.last_kv_cache = None
        # pdb.set_trace()

        if self.device.type == 'xpu':
            torch.xpu.empty_cache()
        st_timestamp = time.perf_counter()
        outputs = self.model.forward(**kwargs)
        # tmp = torch.xpu.memory_stats()
        # logger.info(f"0: {tmp['allocated_bytes.all.current']}")
        # self.last_seq_ids = cur_seq_ids[:]
        # self.last_kv_cache = outputs.past_key_values
        self._set_last_seq_ids(cur_seq_ids[:])
        self._set_last_kv_cache(outputs.past_key_values)
        # pdb.set_trace()

        logits = outputs.logits[:, -1, :]
        bigdl_output = self.sampler(logits, input_metadata, st_timestamp)
        # tmp = torch.xpu.memory_stats()
        # logger.info(f"before: {tmp['allocated_bytes.all.current']}")

        self.update_kv_cache(cur_seq_ids,
                             kv_cache, num_layers, decoder_kv_size)

        # tmp = torch.xpu.memory_stats()
        # logger.info(f"after: {tmp['allocated_bytes.all.current']}")
        return bigdl_output
