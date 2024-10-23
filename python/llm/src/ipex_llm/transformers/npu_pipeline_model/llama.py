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


import numpy as np
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from typing import Sequence
from intel_npu_acceleration_library.backend.factory import NNFactory


class LowBitLlamaLMHead(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        num_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        model_norm_weight,
        vocab_size: int,
        mode: str = "decode",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value
        self.vocab_size = vocab_size

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # define input, the order self.parameter matters
        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

        hidden_states = input

        # model norm and lm head
        model_norm_weight = self.constant(model_norm_weight)
        hidden_states = self.layer_norm(hidden_states, model_norm_weight)
        hidden_states = self.linear(
            hidden_states, self.vocab_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )

        # define outputs
        hidden_states = self.convert_to_fp32(hidden_states)

        print("start compiling")
        self.compile()


class LlamaEmbedding(NNFactory):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        padding_idx,
        dtype,  # fp16
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.dtype = dtype

        # define input
        weight = self.parameter((vocab_size, embedding_dim))
        input = self.parameter((1, 1), dtype=np.int32)

        if padding_idx == -1:
            padding_idx += vocab_size

        if padding_idx is not None:
            masked_embeddings = np.ones(weight.shape, dtype='int64')
            masked_embeddings[padding_idx, :] = 0  # mask

            node_mask = self.constant(masked_embeddings)
            node_masked_w = self.matmul(weight, node_mask, False, True)

        axis_node = self.constant(np.array([0], dtype=np.int64))
        res = self.gather(node_masked_w if padding_idx else weight, input, axis_node, 0)

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()
