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

import os
import torch
import time
import argparse

from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Optional, Sequence, List, Union, Any, Tuple
import numpy as np
import math
from intel_npu_acceleration_library.backend.runtime import set_contiguous, record_function
from intel_npu_acceleration_library.backend.runtime import adapt_output_tensor, _model_cache
from collections import deque
from transformers.cache_utils import Cache
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
import ctypes
from ipex_llm.utils.common import invalidInputError
from typing import Optional, List, Generator
import uuid
from functools import partial
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
from filelock import FileLock

from transformers.utils import logging

logger = logging.get_logger(__name__)
import gc
from colorama import Fore, Back, Style
import torch.multiprocessing as mp
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from ipex_llm.transformers.npu_models.common import reshape_lm_head_input
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss


class LowBitBaichuanMultiDecoderlayer(LLMBaseNNFactory):
    def __init__(
        self,
        # batch_size: int,
        # seq_len: int,
        # hidden_size: int,
        hidden_shape: Sequence[int],
        *shapes,
        num_heads: int,
        # num_key_value_heads: int,
        num_layers: int,
        cached_cos,
        cached_sin,
        input_layernorm_weights=None,
        post_attn_layernorm_weights=None,
        mode: str = "prefill",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        rms_norm_eps,
        intermediate_size,
        n_splits_linear: int = 1,
        n_splits_down_proj: int = 1,
        group_size: int = 0,
        asym: bool = False,
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device,
                         n_splits_linear=n_splits_linear,
                         n_splits_down_proj=n_splits_down_proj,
                         group_size=group_size,
                         asym=asym)
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.cached_cos = cached_cos
        self.cached_sin = cached_sin
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value
        self.num_layers = num_layers
        self.asym = asym

        cos = self.constant(self.cached_cos)
        self.cos = self.unsqueeze(cos, axis=0)

        sin = self.constant(self.cached_sin)
        self.sin = self.unsqueeze(sin, axis=0)

        if mode == "decode":
            self.kv_seq_len = self.max_seq_len + 1
        else:
            self.kv_seq_len = self.seq_len

        self.num_heads = num_heads

        self.head_dim = self.hidden_size // self.num_heads

        # define input, the order self.parameter matters
        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

        # Self Attention
        if mode == "decode":
            attention_mask = self.create_input_op((self.batch_size, 1, 1, self.max_seq_len + 1),
                                                  dtype=np.float16)
        else:
            attention_mask = None

        position_ids = self.create_input_op((self.batch_size, self.seq_len), dtype=np.int64)
        # self.num_key_value_heads = num_key_value_heads

        if input_layernorm_weights is None:
            input_layernorm_weights = []
            post_attn_layernorm_weights = []
            for i in range(num_layers):
                input_layernorm_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
                post_attn_layernorm_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
        else:
            input_layernorm_weights = [self.constant(w) for w in input_layernorm_weights]
            post_attn_layernorm_weights = [self.constant(w) for w in post_attn_layernorm_weights]

        past_keys = []
        past_values = []
        if mode == "decode":
            for i in range(num_layers):
                past_key = self.create_cache_op(
                    (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim)
                )
                if transpose_value:
                    past_value = self.create_cache_op(
                        (self.batch_size, self.num_heads, self.head_dim, self.max_seq_len)
                    )
                else:
                    past_value = self.create_cache_op(
                        (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim)
                    )
                past_keys.append(past_key)
                past_values.append(past_value)
        else:
            past_keys = [None] * num_layers
            past_values = [None] * num_layers

        hidden_states = input

        curr_key_values = []
        for i in range(num_layers):
            hidden_states, new_key_states, new_value_states = self.build_decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_layernorm_weight=input_layernorm_weights[i],
                post_attention_layernorm_weight=post_attn_layernorm_weights[i],
                past_key=past_keys[i],
                past_value=past_values[i],
                use_prefill_sdp=True,
            )
            curr_key_values.append((new_key_states, new_value_states))

        # define outputs
        hidden_states = self.convert_to_fp16(hidden_states)

        for i in range(num_layers):
            new_key_states = self.convert_to_fp16(curr_key_values[i][0])
            new_value_states = self.convert_to_fp16(curr_key_values[i][1])

        print("start compiling")
        if mode == "prefill" and os.environ.get("IPEX_LLM_NPU_DISABLE_COMPILE_OPT", "0") != "1":
            self.compile(npu_dpu_groups=6)
        else:
            self.compile()

    def attention(self,
                  *,
                  hidden_states,
                  position_ids,
                  attention_mask,
                  past_key,
                  past_value,
                  cos,
                  sin,
                  mode,
                  num_heads,
                  head_dim,
                  seq_len,
                  q_bias=None,
                  k_bias=None,
                  v_bias=None,
                  use_prefill_sdp=False):
        hidden_size = num_heads * head_dim
        if self.n_splits_linear != 1:
            hidden_states = self.unsqueeze(hidden_states, axis=0)

        proj = self.linear(
            hidden_states,
            3 * hidden_size,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )

        proj = self.reshape(proj, [-1, 3, hidden_size])  # b*s, 3, h
        proj = self.unsqueeze(proj, [0])  # b, s, 3, h
        proj = self.transpose(proj, [2, 1, 0, 3])  # 3, s, b, h
        proj = self.squeeze(proj)  # 3, b*s, h
        query_states = self.reshape(proj[0, ...], [1, self.seq_len, num_heads, head_dim])
        query_states = self.transpose(query_states, [0, 2, 1, 3])
        key_states = self.reshape(proj[1, ...], [1, self.seq_len, num_heads, head_dim])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        value_states = self.reshape(proj[2, ...], [1, self.seq_len, num_heads, head_dim])

        use_ov_sdp = (mode == "prefill") and use_prefill_sdp
        if self.transpose_value:
            new_value_states = self.transpose(value_states, [0, 2, 3, 1])
            if use_ov_sdp:
                value_states = self.transpose(value_states, [0, 2, 1, 3])
            else:
                value_states = new_value_states
        else:
            value_states = self.transpose(value_states, [0, 2, 1, 3])

        cos = self.unsqueeze(self.squeeze(cos), [0])
        sin = self.unsqueeze(self.squeeze(sin), [0])

        query_states, key_states = self.apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )
        new_key_states = key_states

        if self.mode == "decode":
            key_states = self.concat(past_key, key_states, axis=-2)
            if self.transpose_value:
                value_states = self.concat(past_value, value_states, axis=-1)
            else:
                value_states = self.concat(past_value, value_states, axis=-2)

        if use_ov_sdp:
            value_states = self.convert_to_fp32(value_states)
            key_states = self.convert_to_fp32(key_states)
            query_states = self.convert_to_fp32(query_states)
            attn_output = self.scaled_dot_product_attention(
                query_states, key_states, value_states, None, True)
            attn_output = self.convert_to_fp16(attn_output)
        else:
            attn_weight = self.matmul(query_states, key_states, False, True) / (
                math.sqrt(self.head_dim))
            attn_weight = self.eltwise_add(attn_weight, attention_mask)
            attn_weight = self.convert_to_fp32(attn_weight)
            attn_weight = self.softmax(attn_weight, -1)
            attn_weight = self.convert_to_fp16(attn_weight)
            attn_output = self.matmul(attn_weight, value_states, False, self.transpose_value)

        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(attn_output, [1, seq_len, hidden_size])

        attn_output = self.linear(
            attn_output, hidden_size, hidden_size, bias=False, wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )
        return attn_output, new_key_states, new_value_states

    def build_decoder(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        input_layernorm_weight,
        post_attention_layernorm_weight,
        past_key=None,
        past_value=None,
        use_prefill_sdp=False,
    ):

        residual = hidden_states

        input_2d = self.reshape(hidden_states, (self.batch_size * self.seq_len, self.hidden_size))
        input_2d = self.layer_norm(input_2d, input_layernorm_weight)

        # attention
        attn_output, new_key_states, new_value_states = self.attention(
            hidden_states=input_2d,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key=past_key,
            past_value=past_value,
            cos=self.cos,
            sin=self.sin,
            mode=self.mode,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            seq_len=self.seq_len,
            use_prefill_sdp=use_prefill_sdp,
        )

        hidden_states = self.eltwise_add(residual, attn_output)
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states, post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, self.seq_len, self.mode)
        hidden_states = self.eltwise_add(residual, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)

        return hidden_states, new_key_states, new_value_states


class FusedBaichuanLowBitMultiDecoderlayer(torch.nn.Module):

    def __init__(
        self,
        parameters: List[Tuple[torch.Tensor]],
        input_laynorm_weights: List[torch.Tensor],
        post_attn_layernorm_weights: List[torch.Tensor],
        layer_indexes: List[int],
        intra_stages: int,
        cached_cos: torch.Tensor,
        cached_sin: torch.Tensor,
        num_heads: int,
        head_dim: int,
        # num_key_value_heads: int,
        rms_norm_eps,
        intermediate_size,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        do_print: bool = False,
        n_splits_linear: int = 1,
        n_splits_down_proj: int = 1,
        group_size: int = 0,
        asym: bool = False,
    ):
        super().__init__()

        self.do_print = do_print

        op_parameters = []
        for w in parameters:
            if isinstance(w, tuple) and not asym:  # from QuantizedLinear
                op_parameters.append((w[0].numpy(), w[1].numpy()))
            elif isinstance(w, tuple) and asym:  # from QuantizedLinear
                op_parameters.append((w[0].numpy(), w[1].numpy(),  w[2].numpy()))
            elif w.dtype in [torch.int8, torch.uint8]:    # QuantizedLinear weight
                op_parameters.append(w.numpy())
            elif isinstance(w, np.ndarray):     # scale
                op_parameters.append(w)
            else:
                op_parameters.append(w.to(torch.float16).numpy())
        self.op_parameters = op_parameters
        self.op_id = str(uuid.uuid4())
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        if isinstance(parameters[0], tuple):
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
        elif parameters[0].dtype == torch.int8:
            np_dtype = np.int8
        elif parameters[0].dtype == torch.uint8:
            np_dtype = np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        self.intra_stages = intra_stages
        self.layer_indexes = layer_indexes
        num_layers = len(self.layer_indexes) // intra_stages
        self.layer_ranges = []
        for i in range(intra_stages):
            if i == intra_stages - 1:
                self.layer_ranges.append((i * num_layers, len(self.layer_indexes)))
            else:
                self.layer_ranges.append((i * num_layers, (i + 1) * num_layers))

        self.backend_decoders = []

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            lm_0 = input_laynorm_weights[start:end]
            lm_1 = post_attn_layernorm_weights[start:end]
            decoder = LowBitBaichuanMultiDecoderlayer(
                [1, 1, num_heads * head_dim],
                input_layernorm_weights=lm_0,
                post_attn_layernorm_weights=lm_1,
                cached_cos=cached_cos,
                cached_sin=cached_sin,
                num_heads=num_heads,
                # num_key_value_heads=num_key_value_heads,
                num_layers=end - start,
                max_seq_len=max_seq_len,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                mode="decode",
                transpose_value=self.transpose_value,
                dtype=np_dtype,
                n_splits_linear=n_splits_linear,
                n_splits_down_proj=n_splits_down_proj,
                group_size=group_size,
                asym=asym,
            )
            self.backend_decoders.append(decoder)

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            self.backend_decoders[i].set_weights(self.op_id, op_parameters[start * 5:end * 5])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:

        inputs = (
            hidden_states.to(torch.float16),
            attention_mask.to(torch.float16),
            position_ids.to(torch.int64),
        )

        for i in range(self.intra_stages):
            start, end = self.layer_ranges[i]
            self.backend_decoders[i].update_cache(past_key_value, self.layer_indexes[start:end])

        hidden_states, new_keys, new_values = LowBitBaichuanMultiDecoderlayer.run_decoders(
            inputs,
            decoders=self.backend_decoders)

        if self.do_print:
            print("outputs:", hidden_states)

        outputs = (hidden_states,)
        outputs += (past_key_value, new_keys, new_values)
        return outputs

    def post_forward(self, past_key_value, new_keys, new_values):
        cache_kwargs = {
            "max_seq_len": self.max_seq_len,
            "transpose": self.transpose_value,
        }

        for i in range(len(self.layer_indexes)):
            key_states, value_states = past_key_value.update(
                new_keys[i],
                new_values[i],
                self.layer_indexes[i],
                cache_kwargs,
            )

        for i in range(self.intra_stages):
            self.backend_decoders[i].load_cache_async()


class FusedBaichuanLowBitDecoderlayer(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
        cached_cos,
        cached_sin,
        layer_norm_0,
        layer_norm_1,
        num_heads: int,
        # num_key_value_heads: int,
        layer_idx: int,
        rms_norm_eps,
        intermediate_size,
        max_seq_len: int = 128,
        transpose_value: bool = False,
        n_splits_linear: int = 1,
        n_splits_down_proj: int = 1,
        group_size: int = 0,
        asym: bool = False,
    ):
        super().__init__()
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        # self.rotary_emb = rotary_emb
        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        self.backend_cls_prefill = partial(
            LowBitBaichuanMultiDecoderlayer,
            num_heads=num_heads,
            # num_key_value_heads=num_key_value_heads,
            num_layers=1,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            input_layernorm_weights=None,
            post_attn_layernorm_weights=None,
            max_seq_len=max_seq_len,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            mode="prefill",
            transpose_value=self.transpose_value,
            dtype=np_dtype,
            n_splits_linear=n_splits_linear,
            n_splits_down_proj=n_splits_down_proj,
            group_size=group_size,
            asym=asym
        )
        self.layer_norm_0 = layer_norm_0
        self.layer_norm_1 = layer_norm_1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """

        seq_len = hidden_states.shape[1]

        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16),
                  position_ids.to(torch.int64))
        inputs += (self.layer_norm_0, self.layer_norm_1)
        hidden_states, past_key, past_value = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )
        cache_kwargs = {
            "max_seq_len": self.max_seq_len,
            "transpose": self.transpose_value,
        }
        key_states, value_states = past_key_value.update(
            past_key, past_value, self.layer_idx, cache_kwargs
        )

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs


def run_decode(
    model,
    rank,
    world_size,
    port,
    layer_start,
    layer_end,
    intra_stages,
    max_seq_len,
    transpose_value_cache,
    input_queue,
    result_queue,
):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    print("start init process group, rank: ", rank, "world_size: ", world_size)

    dist.init_process_group()
    my_rank = dist.get_rank()
    my_size = dist.get_world_size()
    logger.info(f"rank: {my_rank}, size: {my_size}")

    num_heads = model.model.layers[layer_start].self_attn.num_heads
    # num_key_value_heads = model.model.layers[layer_start].self_attn.num_key_value_heads
    head_dim = model.model.layers[layer_start].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    intermediate_size = model.config.intermediate_size
    group_size = getattr(model.config, "group_size", 0)
    layer_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    layer_indexs = range(layer_start, layer_end)
    n_splits_linear = len(model.model.layers[0].mlp.gate_proj_dq_list)
    n_splits_down_proj = len(model.model.layers[0].mlp.down_proj_dq_list)
    asym = getattr(model.config, "asym", False)
    for layer_idx in layer_indexs:
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        weights = []
        for layer_list in [attn_layer.W_pack_dq_list, attn_layer.o_proj_dq_list,
                           mlp_layer.gate_proj_dq_list, mlp_layer.up_proj_dq_list,
                           mlp_layer.down_proj_dq_list]:
            l_weights = []
            scales = []
            zeros = []
            for l in layer_list:
                l_weights.append(l.weight)
                scales.append(l.scale)
                if l.zero is not None:
                    zeros.append(l.zero)
            if len(zeros):
                weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0),
                                torch.stack(zeros, axis=0)))
            else:
                weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0)))

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        layer_weights.extend(weights)
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)

    multi_decoder = FusedBaichuanLowBitMultiDecoderlayer(
        parameters=layer_weights,
        input_laynorm_weights=input_layer_norm_weights,
        post_attn_layernorm_weights=post_attn_layernorm_weights,
        layer_indexes=layer_indexs,
        intra_stages=intra_stages,
        cached_cos=cached_cos,
        cached_sin=cached_sin,
        num_heads=num_heads,
        head_dim=head_dim,
        # num_key_value_heads=num_key_value_heads,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        transpose_value=transpose_value_cache,
        do_print=False,
        n_splits_linear=n_splits_linear,
        n_splits_down_proj=n_splits_down_proj,
        group_size=group_size,
        asym=asym,
    )

    dist.barrier()

    past_key_values = None

    control = torch.empty((), dtype=torch.int)
    hidden_states = torch.empty((1, 1, head_dim * num_heads), dtype=torch.float16)
    with torch.inference_mode():
        while True:

            dist.broadcast(control, src=0, async_op=False)
            if control.item() == -2:
                break
            elif control.item() == -1:
                past_key_values = input_queue.get()
            else:
                past_key_values_length = past_key_values.get_seq_length()
                seq_length_with_past = 1 + past_key_values_length
                position_ids = torch.arange(
                    past_key_values_length, seq_length_with_past, dtype=torch.long
                )
                position_ids = position_ids.unsqueeze(0).view(-1, 1)
                attention_mask = torch.ones((1, seq_length_with_past), dtype=torch.bool)
                attention_mask = model.model._prepare_decoder_attention_mask(
                    attention_mask, (1, 1), hidden_states, past_key_values_length
                )

                pad_len = multi_decoder.max_seq_len + 1 - attention_mask.size(-1)

                pad_mask = (0, pad_len)
                padded_causal_mask = F.pad(
                    attention_mask.to(torch.float16), pad_mask, value=torch.finfo(torch.float16).min
                )
                padded_causal_mask[:, :, :, -1] = 0.0
                dist.recv(hidden_states, src=rank - 1)
                layer_outputs = multi_decoder(
                    hidden_states,
                    attention_mask=padded_causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                )
                hidden_states = layer_outputs[0]
                dist.send(hidden_states, dst=(rank + 1) % world_size)
                past_key_values = layer_outputs[1]
                new_keys = layer_outputs[2]
                new_values = layer_outputs[3]
                multi_decoder.post_forward(past_key_values, new_keys, new_values)


class DecodeRunner:
    def __init__(self, model, max_seq_len, intra_pp=2, inter_pp=2, transpose_value_cache=True):
        self.model = model
        self.max_seq_len = max_seq_len
        self.transpose_value_cache = transpose_value_cache
        world_size = inter_pp + 1
        intra_stages = intra_pp
        num_layers = self.model.model.config.num_hidden_layers

        port = "54791"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)

        self.input_queues = []
        self.output_queues = []
        self.decoder_processes = []
        self.forward_signal = torch.tensor(0, dtype=torch.int)

        for rank in range(1, world_size):
            input_q = mp.Queue()
            output_q = mp.Queue()
            start_layer = (rank - 1) * (num_layers // (world_size - 1))
            end_layer = (rank) * (num_layers // (world_size - 1))
            if rank == world_size - 1:
                end_layer = num_layers
            p = mp.Process(
                target=run_decode,
                args=(
                    self.model,
                    rank,
                    world_size,
                    port,
                    start_layer,
                    end_layer,
                    intra_stages,
                    self.max_seq_len,
                    self.transpose_value_cache,
                    input_q,
                    output_q,
                ),
            )
            p.daemon = True
            p.start()
            self.input_queues.append(input_q)
            self.output_queues.append(output_q)
            self.decoder_processes.append(p)

        dist.init_process_group()
        my_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        logger.info(f"rank: {my_rank}, size: {self.world_size}")

        dist.barrier()
        self.cache_past_key_value = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        if self.cache_past_key_value != past_key_value:
            control = torch.tensor(-1, dtype=torch.int)
            dist.broadcast(control, src=0)
            for i in range(len(self.decoder_processes)):
                self.input_queues[i].put(past_key_value)

        dist.broadcast(self.forward_signal, src=0, async_op=True)
        hidden_states = hidden_states.to(torch.float16)
        dist.send(hidden_states, dst=1)
        past_key_value.expand(self.transpose_value_cache)
        dist.recv(hidden_states, src=self.world_size - 1)
        return hidden_states, past_key_value

    def shutdown(self):
        control = torch.tensor(-2, dtype=torch.int)
        dist.broadcast(control, src=0)
        for p in self.decoder_processes:
            p.join(3)
        for p in self.decoder_processes:
            if p.exitcode is None:
                p.kill()

    def __del__(self):
        self.shutdown()


def run_prefill(
    model, max_output_len, max_prompt_len, transpose_value_cache, input_queue, result_queue
):

    layer_start = 0
    layer_end = len(model.model.layers)
    num_heads = model.model.layers[layer_start].self_attn.num_heads
    # num_key_value_heads = model.model.layers[layer_start].self_attn.num_key_value_heads
    head_dim = model.model.layers[layer_start].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    intermediate_size = model.config.intermediate_size
    group_size = getattr(model.config, "group_size", 0)
    deocderlayers = []
    layer_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    layer_indexs = range(layer_start, layer_end)
    n_splits_linear = len(model.model.layers[0].mlp.gate_proj_dq_list)
    n_splits_down_proj = len(model.model.layers[0].mlp.down_proj_dq_list)
    asym = getattr(model.config, "asym", False)
    for layer_idx in layer_indexs:
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        weights = []
        for layer_list in [attn_layer.W_pack_dq_list, attn_layer.o_proj_dq_list,
                           mlp_layer.gate_proj_dq_list, mlp_layer.up_proj_dq_list,
                           mlp_layer.down_proj_dq_list]:
            l_weights = []
            scales = []
            zeros = []
            for l in layer_list:
                l_weights.append(l.weight)
                scales.append(l.scale)
                if l.zero is not None:
                    zeros.append(l.zero)
            if len(zeros):
                weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0),
                                torch.stack(zeros, axis=0)))
            else:
                weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0)))

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)

        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        new_decoderlayer = FusedBaichuanLowBitDecoderlayer(
            weights,
            num_heads=num_heads,
            # num_key_value_heads=num_key_value_heads,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            layer_norm_0=layer_norm_0,
            layer_norm_1=layer_norm_1,
            layer_idx=layer_idx,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            max_seq_len=max_output_len,
            transpose_value=transpose_value_cache,
            n_splits_linear=n_splits_linear,
            n_splits_down_proj=n_splits_down_proj,
            group_size=group_size,
            asym=asym
        )

        layer_weights.extend(weights)
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)
        model.model.layers[layer_idx] = new_decoderlayer
        deocderlayers.append(new_decoderlayer)

    print("finish creating all decode layers in prefill")
    result_queue.put("loading finish")

    while True:

        result = input_queue.get()
        if result == "stop":
            break

        hidden_states, position_ids, causal_mask, past_key_values = result
        with torch.inference_mode():
            for decoder_layer in deocderlayers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    # cache_position=cache_position,
                )

                hidden_states = layer_outputs[0]
                next_decoder_cache = layer_outputs[1]

            result_queue.put((hidden_states, next_decoder_cache))


class PrefillRunner:
    def __init__(self, model, max_output_len, max_prompt_len, transpose_value_cache):
        self.model = model
        self.max_output_len = max_output_len
        self.max_prompt_len = max_prompt_len
        self.transpose_value_cache = transpose_value_cache

        self.prefill_result_queue = mp.Queue()
        self.prefill_input_queue = mp.Queue()

        self.p = mp.Process(
            target=run_prefill,
            args=(
                model,
                max_output_len,
                max_prompt_len,
                transpose_value_cache,
                self.prefill_input_queue,
                self.prefill_result_queue,
            ),
        )
        self.p.daemon = True
        self.p.start()
        output = self.prefill_result_queue.get()
        print(Fore.GREEN + f"prefill process output: {output}")
        print(Style.RESET_ALL)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        seq_len = hidden_states.size(1)
        invalidInputError(
            seq_len <= self.max_prompt_len,
            (
                f"seq_len: {seq_len} should be less than or equal"
                f" to max_prompt_len {self.max_prompt_len}"
            ),
        )
        pad_len = self.max_prompt_len - seq_len
        hidden_states = F.pad(hidden_states.to(torch.float16), (0, 0, 0, pad_len), value=0.0)
        position_ids = F.pad(position_ids, (0, pad_len), value=0)
        attention_mask = F.pad(
            attention_mask.to(torch.float16),
            (0, pad_len, 0, pad_len),
            value=torch.finfo(torch.float16).min,
        )

        args = (hidden_states, position_ids, attention_mask, past_key_value)
        self.prefill_input_queue.put(args)
        hidden_states, past_key_value = self.prefill_result_queue.get()
        past_key_value.shrink(seq_len, self.transpose_value_cache)
        hidden_states = hidden_states[:, :seq_len, :]
        return hidden_states, past_key_value

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


def gen_baichuan_fused_model_forward(prefill_runner, decode_runner):
    def baichuan_fused_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            invalidInputError(False, "You cannot specify both decoder_input_ids\
                              and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            invalidInputError(False, "You have to specify either decoder_input_ids\
                              or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # ipex-llm changes start
        from ipex_llm.transformers.npu_models.kv import DynamicFusedNormalCache

        if use_cache and not isinstance(past_key_values, DynamicFusedNormalCache):
            past_key_values = DynamicFusedNormalCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing.\
                        Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if seq_length == 1:
            layers_runner = decode_runner
        else:
            layers_runner = prefill_runner
        layer_outputs = layers_runner.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ipex-llm changes start
        next_cache = next_decoder_cache if use_cache else None
        # ipex-llm changes end
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return baichuan_fused_model_forward


def baichuan2_causal_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    # ipex-llm change start
    hidden_states = reshape_lm_head_input(hidden_states)
    # ipex-llm change end
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        softmax_normalizer = shift_logits.max(-1).values ** 2
        z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) + z_loss

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
