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
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import List, Union, Any
from intel_npu_acceleration_library.backend.runtime import set_contiguous, record_function
from intel_npu_acceleration_library.backend.runtime import adapt_output_tensor, _model_cache
from collections import deque
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from ipex_llm.utils.common import invalidInputError
from transformers.utils import logging
from filelock import FileLock
import ctypes
import math
import numpy as np
import torch.multiprocessing as mp
from colorama import Fore, Back, Style
from typing import Optional, Sequence, List, Union, Any, Tuple
from transformers.cache_utils import Cache
import uuid
from functools import partial
import torch.distributed as dist
import torch.nn.functional as F


logger = logging.get_logger(__name__)


@torch.no_grad()
def run_model(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: str,
    replica: int = 1,
) -> torch.Tensor:
    global _model_cache
    import time

    t0 = time.perf_counter()

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    if not isinstance(x, (list, tuple)):
        x = [x]

    # Reshape input
    input_dtype = x[0].dtype
    x_np = [set_contiguous(elem).to(torch.float16).numpy() for elem in x]
    op_args = []
    op_args_flatten = []
    for w in weights:
        if isinstance(w, tuple):  # from QuantizedLinear
            op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy()))
            op_args_flatten.append(op_args[-1][0])
            op_args_flatten.append(op_args[-1][1])
        else:
            op_args.append(set_contiguous(w).to(torch.float16).numpy())
            op_args_flatten.append(op_args[-1])

    shape_dtype_signature = "_".join(
        ["_".join(str(dim) for dim in t.shape) + f"_{t.dtype}" for t in x_np + op_args_flatten]
    )
    key = f"{backend_cls.func.__name__}_{shape_dtype_signature}"
    models = _model_cache.get(key, None)

    input_shapes = [elem.shape for elem in x_np]
    if models is None:
        _model_cache[key] = deque([backend_cls(*input_shapes) for i in range(replica)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(*input_shapes))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    with record_function(f"npu_factory_mul_{key}"):
        ret = model.run(x_np, *op_args, **op_kwargs)

    if isinstance(ret, list):
        results = [adapt_output_tensor(r, r.shape, input_dtype) for r in ret]
    else:
        results = adapt_output_tensor(ret, ret.shape, input_dtype)

    return results


class LLMBaseNNFactory(NNFactory):

    def __init__(self, max_seq_len, transpose_value, dtype, profile=False, device="NPU"):
        super().__init__(profile, device)
        self.cache_parameter_ops = []
        self.input_ops = []
        self.linear_ops = []
        self.kv_cache_c_handle = None
        self.kv_cache_torch = []
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        self.dtype = dtype

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
                  num_key_value_heads,
                  head_dim,
                  seq_len,
                  q_bias=None,
                  k_bias=None,
                  v_bias=None):
        hidden_size = num_heads * head_dim
        num_key_value_groups = num_heads // num_key_value_heads
        query_states = self.linear(
            hidden_states,
            num_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        if q_bias is not None:
            query_states = query_states + q_bias
        key_states = self.linear(
            hidden_states,
            num_key_value_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        if k_bias is not None:
            key_states = key_states + k_bias
        value_states = self.linear(
            hidden_states,
            num_key_value_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        if v_bias is not None:
            value_states = value_states + v_bias

        query_states = self.reshape(
            query_states, [1, seq_len, num_heads, head_dim]
        )
        key_states = self.reshape(
            key_states, [1, seq_len, num_key_value_heads, head_dim]
        )
        value_states = self.reshape(
            value_states, [1, seq_len, num_key_value_heads, head_dim]
        )

        query_states = self.transpose(query_states, [0, 2, 1, 3])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        if self.transpose_value:
            value_states = self.transpose(value_states, [0, 2, 3, 1])
        else:
            value_states = self.transpose(value_states, [0, 2, 1, 3])

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
        new_value_states = value_states

        if mode == "decode":
            key_states = self.concat(past_key, key_states, axis=-2)
            if self.transpose_value:
                value_states = self.concat(past_value, value_states, axis=-1)
            else:
                value_states = self.concat(past_value, value_states, axis=-2)
            kv_seq_len = self.max_seq_len + 1
        else:
            kv_seq_len = seq_len

        key_states = self.repeat_kv(hidden_states=key_states,
                                    n_rep=num_key_value_groups,
                                    num_key_value_heads=num_key_value_heads,
                                    kv_seq_len=kv_seq_len,
                                    head_dim=head_dim,)
        value_states = self.repeat_kv(hidden_states=value_states,
                                      n_rep=num_key_value_groups,
                                      num_key_value_heads=num_key_value_heads,
                                      kv_seq_len=kv_seq_len,
                                      head_dim=head_dim,
                                      transpose=self.transpose_value)
        attn_weight = self.matmul(query_states, key_states, False, True) / (
            math.sqrt(head_dim)
        )
        attn_weight = self.eltwise_add(attn_weight, attention_mask)
        attn_weight = self.convert_to_fp32(attn_weight)
        attn_weight = self.softmax(attn_weight, -1)
        attn_weight = self.convert_to_fp16(attn_weight)
        attn_output = self.matmul(attn_weight, value_states, False, self.transpose_value)

        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(attn_output, [1, seq_len, hidden_size])

        attn_output = self.linear(
            attn_output, hidden_size, hidden_size, bias=False, wt_dtype=self.dtype
        )

        return attn_output, new_key_states, new_value_states

    def mlp(self, hidden_states):
        mm1 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )
        mm2 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )  # type: ignore[attr-defined]
        mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
        hidden_states = self.linear(
            mm1, self.hidden_size, self.intermediate_size, bias=False, wt_dtype=self.dtype
        )
        return hidden_states

    def layer_norm(self, hidden_states, layernorm_weight):
        hidden_states = self.convert_to_fp32(hidden_states)
        variance = self.reduce_mean(
            self.power(hidden_states, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(self.rms_norm_eps)
        hidden_states = self.eltwise_div(hidden_states, self.sqrt(self.eltwise_add(variance, eps)))
        layernorm_weight = self.convert_to_fp32(layernorm_weight)
        hidden_states = self.eltwise_mul(layernorm_weight, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)
        return hidden_states

    def rotate_half(self, x, *, num_heads, seq_len, head_dim):
        x1 = self.slice(
            x,
            [0, 0, 0, 0],
            [1, num_heads, seq_len, head_dim // 2],
        )
        x2 = self.slice(
            x,
            [0, 0, 0, head_dim // 2],
            [1, num_heads, seq_len, head_dim],
        )
        return self.concat(self.negative(x2), x1, axis=-1)

    def apply_rotary_pos_emb(self, *, q, k, cos, sin, position_ids,
                             num_heads, seq_len, head_dim):
        position_ids = self.squeeze(position_ids)
        cos = self.gather(cos, self.convert_to_int32(position_ids), self.constant(1), 0)
        sin = self.gather(sin, self.convert_to_int32(position_ids), self.constant(1), 0)
        cos = self.unsqueeze(cos, [1])
        sin = self.unsqueeze(sin, [1])

        rotate_half_q = self.rotate_half(q,
                                         num_heads=num_heads,
                                         seq_len=seq_len,
                                         head_dim=head_dim)
        rotate_half_k = self.rotate_half(k,
                                         num_heads=num_heads,
                                         seq_len=seq_len,
                                         head_dim=head_dim)

        q_embed = self.eltwise_add(
            self.eltwise_mul(q, cos), self.eltwise_mul(rotate_half_q, sin)
        )
        k_embed = self.eltwise_add(
            self.eltwise_mul(k, cos), self.eltwise_mul(rotate_half_k, sin)
        )

        return q_embed, k_embed

    def repeat_kv(self, *, hidden_states, n_rep, num_key_value_heads,
                  kv_seq_len, head_dim, transpose=False):
        if n_rep == 1:
            return hidden_states
        if not transpose:
            hidden_states = self.reshape(
                hidden_states,
                [1, num_key_value_heads, 1, kv_seq_len, head_dim],
            )
            hidden_states = self.broadcast(
                hidden_states,
                [1, num_key_value_heads, n_rep, kv_seq_len, head_dim],
            )
            hidden_states = self.reshape(
                hidden_states,
                [1, n_rep * num_key_value_heads, kv_seq_len, head_dim],
            )
        else:
            hidden_states = self.reshape(
                hidden_states,
                [1, num_key_value_heads, 1, head_dim, kv_seq_len],
            )
            hidden_states = self.broadcast(
                hidden_states,
                [1, num_key_value_heads, n_rep, head_dim, kv_seq_len],
            )
            hidden_states = self.reshape(
                hidden_states,
                [1, n_rep * num_key_value_heads, head_dim, kv_seq_len],
            )
        return hidden_states

    def create_cache_op(self, shape):
        invalidInputError(len(self.linear_ops) == 0,
                          "create_cache_op should be called before any linear op")
        op = super().parameter(shape)
        self.cache_parameter_ops.append(op)
        return op

    def create_input_op(self, shape):
        invalidInputError(len(self.cache_parameter_ops) == 0,
                          "create_input_op should be called before any create_cache_op")
        invalidInputError(len(self.linear_ops) == 0,
                          "create_input_op should be called before any linear op")

        op = super().parameter(shape)
        self.input_ops.append(op)
        return op

    def linear(self, *args, **kwargs):
        op = super().linear(*args, **kwargs)
        self.linear_ops.append(op)
        return op

    def parameter(self, shape):
        invalidInputError(False,
                          ("parameter should not be called directly, "
                           "use create_cache_op or create_input_op instead"))

    def update_cache(self, past_key_value, indexes):

        if self.kv_cache_c_handle is not None:
            curr_ptr = self.kv_cache_torch[0].storage().data_ptr()
            new_ptr = past_key_value.key_cache[indexes[0]].storage().data_ptr()
            if curr_ptr != new_ptr:
                backend_lib.destroyParameters(self.kv_cache_c_handle)
                self.kv_cache_c_handle = None
                self.kv_cache_torch = []
        if self.kv_cache_c_handle is None:
            for idx in indexes:
                past_key = past_key_value.key_cache[idx]
                past_value = past_key_value.value_cache[idx]
                invalidInputError(
                    past_key.dtype == torch.float16, f"past_key dtype is {past_key.dtype}"
                )
                new_size = (past_key.size(0), past_key.size(1), self.max_seq_len, past_key.size(3))
                past_key = past_key.as_strided(new_size, past_key.stride(), storage_offset=0)
                invalidInputError(past_key.is_contiguous(), "past_key is not contiguous")
                past_value = past_value.as_strided(new_size, past_value.stride(), storage_offset=0)
                if self.transpose_value:
                    past_value = past_value.transpose(-1, -2)
                invalidInputError(past_value.is_contiguous(), "past_value is not contiguous")

                self.kv_cache_torch.append(past_key)
                self.kv_cache_torch.append(past_value)

            layer_kv_cache_np = [p.numpy() for p in self.kv_cache_torch]
            invalidInputError(len(self.cache_parameter_ops) == len(layer_kv_cache_np),
                              (f"kv_cache size does not match graph, "
                               f"with kv_cache size: {len(layer_kv_cache_np)} and"
                               f" graph size: {len(self.cache_parameter_ops)}")
                              )
            self.kv_cache_c_handle = self.create_parameters(layer_kv_cache_np)
            self.load_cache_async()

    def load_cache_async(self):
        self.load_wt_fn(len(self.input_ops), self._mm, self.kv_cache_c_handle)

    def set_weights(self, op_id, weights):
        self.set_weights_async(op_id, weights)
        with FileLock(f"decoder_run.lock"):
            backend_lib.run(self._mm)

    def set_weights_async(self, op_id, weights):
        offset = len(self.input_ops) + len(self.cache_parameter_ops)
        invalidInputError(len(weights) == len(self.linear_ops),
                          (f"weights size does not match graph, "
                           f"with weights size: {len(weights)} and "
                           f" graph linear size: {len(self.linear_ops)}"))
        self.setWeights(offset, op_id, *weights)

    @staticmethod
    def run_decoders(inputs, decoders, models_ptr=None):
        x_np = [elem.to(torch.float16).numpy() for elem in inputs]

        num_decoders = len(decoders)
        num_inputs = len(x_np)

        if models_ptr is None:
            array_type = ctypes.POINTER(ctypes.c_char) * num_decoders
            models_ptr = array_type(
                *[decoders[i]._mm for i in range(num_decoders)]
            )

        inputs_ptr = (ctypes.c_void_p * num_inputs)(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in x_np]
        )
        backend_lib.run_decoders(models_ptr, inputs_ptr, num_decoders, num_inputs)

        hidden_states = decoders[-1].torch_out[0]
        new_key_states = []
        new_value_states = []
        for i in range(num_decoders):
            for j in range(1, len(decoders[i].torch_out)):
                if j % 2 == 1:
                    new_key_states.append(decoders[i].torch_out[j])
                else:
                    new_value_states.append(decoders[i].torch_out[j])
        return hidden_states, new_key_states, new_value_states


class qwen:
    class LowBitQwenMultiDecoderlayer(LLMBaseNNFactory):
        def __init__(
            self,
            # batch_size: int,
            # seq_len: int,
            # hidden_size: int,
            hidden_shape: Sequence[int],
            *shapes,
            num_heads: int,
            num_key_value_heads: int,
            num_layers: int,
            cached_cos,
            cached_sin,
            input_layernorm_weights=None,
            post_attn_layernorm_weights=None,
            q_biases=None,
            k_biases=None,
            v_biases=None,
            mode: str = "prefill",
            dtype: np.dtype = np.int8,
            max_seq_len: int = 1024,
            transpose_value: bool = False,
            profile: bool = False,
            device: str = "NPU",
            rms_norm_eps,
            intermediate_size,
        ):
            super().__init__(max_seq_len=max_seq_len,
                             transpose_value=transpose_value,
                             dtype=dtype,
                             profile=profile,
                             device=device)
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

            cos = self.constant(self.cached_cos)
            self.cos = self.unsqueeze(cos, axis=0)

            sin = self.constant(self.cached_sin)
            self.sin = self.unsqueeze(sin, axis=0)

            if mode == "decode":
                self.kv_seq_len = self.max_seq_len + 1
            else:
                self.kv_seq_len = self.seq_len

            self.num_heads = num_heads
            self.num_key_value_heads = num_key_value_heads

            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads

            # define input, the order self.parameter matters
            input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

            # Self Attention
            if mode == "decode":
                attention_mask = self.create_input_op((self.batch_size, 1, 1, self.max_seq_len + 1))
            else:
                attention_mask = self.create_input_op((self.batch_size, 1,
                                                       self.seq_len, self.seq_len))

            position_ids = self.create_input_op((self.batch_size, self.seq_len))
            past_keys = []
            past_values = []
            if mode == "decode":
                for i in range(num_layers):
                    past_key = self.create_cache_op(
                        (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
                    )
                    if transpose_value:
                        past_value = self.create_cache_op(
                            (self.batch_size, self.num_key_value_heads,
                             self.head_dim, self.max_seq_len)
                        )
                    else:
                        past_value = self.create_cache_op(
                            (self.batch_size, self.num_key_value_heads,
                             self.max_seq_len, self.head_dim)
                        )
                    past_keys.append(past_key)
                    past_values.append(past_value)
            else:
                past_keys = [None] * num_layers
                past_values = [None] * num_layers

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
                post_attn_layernorm_weights = \
                    [self.constant(w) for w in post_attn_layernorm_weights]

            if q_biases is None:
                q_biases = []
                k_biases = []
                v_biases = []
                for i in range(num_layers):
                    q_biases.append(self.create_input_op((self.num_heads * self.head_dim,)))
                    k_biases.append(self.create_input_op(
                        (self.num_key_value_heads * self.head_dim,)))
                    v_biases.append(self.create_input_op(
                        (self.num_key_value_heads * self.head_dim,)))
            else:
                q_biases = [self.constant(w) for w in q_biases]
                k_biases = [self.constant(w) for w in k_biases]
                v_biases = [self.constant(w) for w in v_biases]

            hidden_states = input

            curr_key_values = []
            for i in range(num_layers):
                hidden_states, new_key_states, new_value_states = self.build_decoder(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    input_layernorm_weight=input_layernorm_weights[i],
                    post_attention_layernorm_weight=post_attn_layernorm_weights[i],
                    q_bias=q_biases[i],
                    k_bias=k_biases[i],
                    v_bias=v_biases[i],
                    past_key=past_keys[i],
                    past_value=past_values[i],
                )
                curr_key_values.append((new_key_states, new_value_states))

            # define outputs
            hidden_states = self.convert_to_fp16(hidden_states)

            for i in range(num_layers):
                new_key_states = self.convert_to_fp16(curr_key_values[i][0])
                new_value_states = self.convert_to_fp16(curr_key_values[i][1])

            print("start compiling")
            self.compile()
            print("end compiling")

        def mlp(self, hidden_states, seq_len):
            mm1 = self.linear(
                hidden_states, self.intermediate_size,
                self.hidden_size, bias=False, wt_dtype=self.dtype
            )
            mm2 = self.linear(
                hidden_states, self.intermediate_size,
                self.hidden_size, bias=False, wt_dtype=self.dtype
            )  # type: ignore[attr-defined]
            mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
            if self.intermediate_size == 18944:
                # for qwen2-7b
                mm1_0 = self.slice(mm1, begin=[0, 0, 0], end=[1, seq_len, 9472])
                mm1_1 = self.slice(mm1, begin=[0, 0, 9472], end=[1, seq_len, 18944])
                hidden_states_0 = self.linear(mm1_0, self.hidden_size, 9472,
                                              bias=False, wt_dtype=self.dtype)
                hidden_states_1 = self.linear(mm1_1, self.hidden_size, 9472,
                                              bias=False, wt_dtype=self.dtype)
                hidden_states = hidden_states_0 + hidden_states_1
            else:
                hidden_states = self.linear(
                    mm1, self.hidden_size, self.intermediate_size, bias=False, wt_dtype=self.dtype
                )
            return hidden_states

        def build_decoder(
            self,
            hidden_states,
            attention_mask,
            position_ids,
            input_layernorm_weight,
            post_attention_layernorm_weight,
            q_bias,
            k_bias,
            v_bias,
            past_key=None,
            past_value=None,
        ):

            residual = hidden_states
            input_2d = self.reshape(hidden_states,
                                    (self.batch_size * self.seq_len, self.hidden_size))
            input_2d = self.layer_norm(input_2d, input_layernorm_weight)
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
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                seq_len=self.seq_len,
                q_bias=q_bias,
                k_bias=k_bias,
                v_bias=v_bias,
            )
            hidden_states = self.eltwise_add(residual, attn_output)
            residual = hidden_states
            hidden_states = self.layer_norm(hidden_states, post_attention_layernorm_weight)
            hidden_states = self.mlp(hidden_states, self.seq_len)
            hidden_states = self.eltwise_add(residual, hidden_states)
            hidden_states = self.convert_to_fp16(hidden_states)

            return hidden_states, new_key_states, new_value_states

    class FusedQwenLowBitMultiDecoderlayer(torch.nn.Module):
        def __init__(
            self,
            parameters: List[Tuple[torch.Tensor]],
            input_laynorm_weights: List[torch.Tensor],
            post_attn_layernorm_weights: List[torch.Tensor],
            q_biases: List[torch.Tensor],
            k_biases: List[torch.Tensor],
            v_biases: List[torch.Tensor],
            layer_indexes: List[int],
            intra_stages: int,
            cached_cos: torch.Tensor,
            cached_sin: torch.Tensor,
            num_heads: int,
            head_dim: int,
            num_key_value_heads: int,
            rms_norm_eps,
            intermediate_size,
            max_seq_len: int = 1024,
            transpose_value: bool = False,
            do_print: bool = False,
        ):
            super().__init__()

            self.do_print = do_print

            op_parameters = []
            for w in parameters:
                if isinstance(w, tuple):  # from QuantizedLinear
                    op_parameters.append((w[0].numpy(), w[1].numpy()))
                else:
                    op_parameters.append(w.to(torch.float16).numpy())
            self.op_parameters = op_parameters
            self.op_id = str(uuid.uuid4())
            self.max_seq_len = max_seq_len
            self.transpose_value = transpose_value
            if isinstance(parameters[0], tuple):
                np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
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
                decoder = qwen.LowBitQwenMultiDecoderlayer(
                    [1, 1, num_heads * head_dim],
                    input_layernorm_weights=lm_0,
                    post_attn_layernorm_weights=lm_1,
                    q_biases=q_biases[start:end],
                    k_biases=k_biases[start:end],
                    v_biases=v_biases[start:end],
                    cached_cos=cached_cos,
                    cached_sin=cached_sin,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    num_layers=end - start,
                    max_seq_len=max_seq_len,
                    rms_norm_eps=rms_norm_eps,
                    intermediate_size=intermediate_size,
                    mode="decode",
                    transpose_value=self.transpose_value,
                    dtype=np_dtype,
                )
                self.backend_decoders.append(decoder)

            offset = 0
            for i in range(intra_stages):
                start, end = self.layer_ranges[i]
                curr_linear_ops = len(self.backend_decoders[i].linear_ops)
                curr_parameters = self.op_parameters[offset:offset + curr_linear_ops]
                self.backend_decoders[i].set_weights(self.op_id, curr_parameters)
                offset = offset + curr_linear_ops

            array_type = ctypes.POINTER(ctypes.c_char) * intra_stages
            self.models_ptr = array_type(*[self.backend_decoders[i]._mm
                                           for i in range(intra_stages)])

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ) -> torch.Tensor:

            inputs = (
                hidden_states.to(torch.float16),
                attention_mask,
                position_ids,
            )

            for i in range(self.intra_stages):
                start, end = self.layer_ranges[i]
                self.backend_decoders[i].update_cache(past_key_value, self.layer_indexes[start:end])

            hidden_states, new_keys, new_values = qwen.LowBitQwenMultiDecoderlayer.run_decoders(
                inputs,
                self.backend_decoders,
                self.models_ptr)

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

    class FusedQwenLowBitDecoderlayer(torch.nn.Module):
        def __init__(
            self,
            parameters: List[torch.Tensor],
            cached_cos,
            cached_sin,
            layer_norm_0,
            layer_norm_1,
            q_bias,
            k_bias,
            v_bias,
            num_heads: int,
            num_key_value_heads: int,
            layer_idx: int,
            rms_norm_eps,
            intermediate_size,
            max_seq_len: int = 128,
            transpose_value: bool = False,
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
                qwen.LowBitQwenMultiDecoderlayer,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
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
            )
            self.layer_norm_0 = layer_norm_0
            self.layer_norm_1 = layer_norm_1
            self.q_bias = q_bias
            self.k_bias = k_bias
            self.v_bias = v_bias

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            """Torch module forward method.

            Args:
                x (torch.Tensor): Input tensor

            Returns:
                torch.Tensor: result
            """

            seq_len = hidden_states.shape[1]

            backend_cls = self.backend_cls_prefill
            inputs = (hidden_states.to(torch.float16), attention_mask, position_ids)
            inputs += (self.layer_norm_0, self.layer_norm_1)
            inputs += (self.q_bias, self.k_bias, self.v_bias)
            hidden_states, past_key, past_value = run_model(
                inputs, self.op_parameters, backend_cls, self.op_id, replica=2
            )
            cache_kwargs = {"max_seq_len": self.max_seq_len, "transpose": self.transpose_value}
            key_states, value_states = past_key_value.update(
                past_key, past_value, self.layer_idx, cache_kwargs
            )

            outputs = (hidden_states,)
            outputs += (past_key_value,)
            return outputs


class llama:
    class LowBitLlamaMultiDecoderlayer(LLMBaseNNFactory):
        def __init__(
            self,
            # batch_size: int,
            # seq_len: int,
            # hidden_size: int,
            hidden_shape: Sequence[int],
            *shapes,
            num_heads: int,
            num_key_value_heads: int,
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
        ):
            super().__init__(max_seq_len=max_seq_len,
                             transpose_value=transpose_value,
                             dtype=dtype,
                             profile=profile,
                             device=device)
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

            cos = self.constant(self.cached_cos)
            self.cos = self.unsqueeze(cos, axis=0)

            sin = self.constant(self.cached_sin)
            self.sin = self.unsqueeze(sin, axis=0)

            if mode == "decode":
                self.kv_seq_len = self.max_seq_len + 1
            else:
                self.kv_seq_len = self.seq_len

            self.num_heads = num_heads
            self.num_key_value_heads = num_key_value_heads

            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads

            # define input, the order self.parameter matters
            input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

            # Self Attention
            if mode == "decode":
                attention_mask = self.create_input_op((self.batch_size, 1, 1, self.max_seq_len + 1))
            else:
                attention_mask = self.create_input_op(
                    (self.batch_size, 1, self.seq_len, self.seq_len))

            position_ids = self.create_input_op((self.batch_size, self.seq_len))
            past_keys = []
            past_values = []
            if mode == "decode":
                for i in range(num_layers):
                    past_key = self.create_cache_op(
                        (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
                    )
                    if transpose_value:
                        past_value = self.create_cache_op(
                            (self.batch_size, self.num_key_value_heads,
                             self.head_dim, self.max_seq_len)
                        )
                    else:
                        past_value = self.create_cache_op(
                            (self.batch_size, self.num_key_value_heads,
                             self.max_seq_len, self.head_dim)
                        )
                    past_keys.append(past_key)
                    past_values.append(past_value)
            else:
                past_keys = [None] * num_layers
                past_values = [None] * num_layers

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
                post_attn_layernorm_weights = [self.constant(w)
                                               for w in post_attn_layernorm_weights]

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
                )
                curr_key_values.append((new_key_states, new_value_states))

            # define outputs
            hidden_states = self.convert_to_fp16(hidden_states)

            for i in range(num_layers):
                new_key_states = self.convert_to_fp16(curr_key_values[i][0])
                new_value_states = self.convert_to_fp16(curr_key_values[i][1])

            print("start compiling")
            self.compile()

        def build_decoder(
            self,
            hidden_states,
            attention_mask,
            position_ids,
            input_layernorm_weight,
            post_attention_layernorm_weight,
            past_key=None,
            past_value=None,
        ):

            residual = hidden_states
            input_2d = self.reshape(hidden_states,
                                    (self.batch_size * self.seq_len, self.hidden_size))
            input_2d = self.layer_norm(input_2d, input_layernorm_weight)
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
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                seq_len=self.seq_len,
            )
            hidden_states = self.eltwise_add(residual, attn_output)
            residual = hidden_states
            hidden_states = self.layer_norm(hidden_states, post_attention_layernorm_weight)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.eltwise_add(residual, hidden_states)
            hidden_states = self.convert_to_fp16(hidden_states)

            return hidden_states, new_key_states, new_value_states
        
    class FusedLlamaLowBitMultiDecoderlayer(torch.nn.Module):
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
            num_key_value_heads: int,
            rms_norm_eps,
            intermediate_size,
            max_seq_len: int = 1024,
            transpose_value: bool = False,
            do_print: bool = False,
        ):
            super().__init__()

            self.do_print = do_print

            op_parameters = []
            for w in parameters:
                if isinstance(w, tuple):  # from QuantizedLinear
                    op_parameters.append((w[0].numpy(), w[1].numpy()))
                else:
                    op_parameters.append(w.to(torch.float16).numpy())
            self.op_parameters = op_parameters
            self.op_id = str(uuid.uuid4())
            self.max_seq_len = max_seq_len
            self.transpose_value = transpose_value
            if isinstance(parameters[0], tuple):
                np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
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
                decoder = llama.LowBitLlamaMultiDecoderlayer(
                    [1, 1, num_heads * head_dim],
                    input_layernorm_weights=lm_0,
                    post_attn_layernorm_weights=lm_1,
                    cached_cos=cached_cos,
                    cached_sin=cached_sin,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    num_layers=end - start,
                    max_seq_len=max_seq_len,
                    rms_norm_eps=rms_norm_eps,
                    intermediate_size=intermediate_size,
                    mode="decode",
                    transpose_value=self.transpose_value,
                    dtype=np_dtype,
                )
                self.backend_decoders.append(decoder)

            for i in range(intra_stages):
                start, end = self.layer_ranges[i]
                self.backend_decoders[i].set_weights(self.op_id, op_parameters[start * 7:end * 7])

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> torch.Tensor:

            inputs = (
                hidden_states.to(torch.float16),
                attention_mask,
                position_ids,
            )

            for i in range(self.intra_stages):
                start, end = self.layer_ranges[i]
                self.backend_decoders[i].update_cache(past_key_value, self.layer_indexes[start:end])

            hidden_states, new_keys, new_values = llama.LowBitLlamaMultiDecoderlayer.run_decoders(
                inputs,
                decoders=self.backend_decoders)

            if self.do_print:
                print("outputs:", hidden_states)

            outputs = (hidden_states,)
            outputs += (past_key_value, new_keys, new_values)
            return outputs

        def post_forward(self, past_key_value, new_keys, new_values, cache_position):
            cache_kwargs = {
                "cache_position": cache_position,
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

    class FusedLlamaLowBitDecoderlayer(torch.nn.Module):
        """LLAMA MLP operation NPU backend."""

        def __init__(
            self,
            parameters: List[torch.Tensor],
            cached_cos,
            cached_sin,
            layer_norm_0,
            layer_norm_1,
            num_heads: int,
            num_key_value_heads: int,
            layer_idx: int,
            rms_norm_eps,
            intermediate_size,
            max_seq_len: int = 128,
            transpose_value: bool = False,
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
                llama.LowBitLlamaMultiDecoderlayer,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
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
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            """Torch module forward method.

            Args:
                x (torch.Tensor): Input tensor

            Returns:
                torch.Tensor: result
            """

            seq_len = hidden_states.shape[1]

            backend_cls = self.backend_cls_prefill
            inputs = (hidden_states.to(torch.float16), attention_mask, position_ids)
            inputs += (self.layer_norm_0, self.layer_norm_1)
            hidden_states, past_key, past_value = run_model(
                inputs, self.op_parameters, backend_cls, self.op_id, replica=2
            )
            cache_kwargs = {
                "cache_position": cache_position,
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
    model_type = model.config.model_type

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
    num_key_value_heads = model.model.layers[layer_start].self_attn.num_key_value_heads
    head_dim = model.model.layers[layer_start].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    intermediate_size = model.config.intermediate_size
    layer_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    if model_type == "qwen2":
        q_biases = []
        k_biases = []
        v_biases = []
    layer_indexs = range(layer_start, layer_end)
    for layer_idx in layer_indexs:
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        if model_type == "qwen2" and model.config.intermediate_size == 8960:
            # for qwen2-1.5b
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale),
            ]

        elif model_type == "qwen2" and model.config.intermediate_size == 18944:
            # for qwen2-7b
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj_0.weight, mlp_layer.down_proj_0.scale),
                (mlp_layer.down_proj_1.weight, mlp_layer.down_proj_1.scale)
            ]

        elif model_type == "llama":
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale),
            ]

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        layer_weights.extend(weights)
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)

        if model_type == "qwen2":
            q_biases.append(attn_layer.q_proj.bias.to(torch.float16))
            k_biases.append(attn_layer.k_proj.bias.to(torch.float16))
            v_biases.append(attn_layer.v_proj.bias.to(torch.float16))

    if model_type == "llama":
        multi_decoder = llama.FusedLlamaLowBitMultiDecoderlayer(
            parameters=layer_weights,
            input_laynorm_weights=input_layer_norm_weights,
            post_attn_layernorm_weights=post_attn_layernorm_weights,
            layer_indexes=layer_indexs,
            intra_stages=intra_stages,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            transpose_value=transpose_value_cache,
            do_print=False,
        )
    elif model_type == "qwen2":
        multi_decoder = qwen.FusedQwenLowBitMultiDecoderlayer(
            parameters=layer_weights,
            input_laynorm_weights=input_layer_norm_weights,
            post_attn_layernorm_weights=post_attn_layernorm_weights,
            q_biases=q_biases,
            k_biases=k_biases,
            v_biases=v_biases,
            layer_indexes=layer_indexs,
            intra_stages=intra_stages,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            transpose_value=transpose_value_cache,
            do_print=False,
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
                if model_type == "llama":
                    past_seen_tokens = past_key_values.get_seq_length()
                    attention_mask = torch.ones([1, past_seen_tokens + 1], dtype=torch.int64)
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + 1, device=hidden_states.device
                    )

                    position_ids = position_ids = cache_position.unsqueeze(0)
                    causal_mask = model.model._update_causal_mask(
                        attention_mask, hidden_states, cache_position, past_seen_tokens
                    )
                    pad_len = multi_decoder.max_seq_len + 1 - causal_mask.size(-1)

                    pad_mask = (0, pad_len)
                    padded_causal_mask = F.pad(
                        causal_mask.to(torch.float16), pad_mask,
                        value=torch.finfo(torch.float16).min
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
                        cache_position=cache_position,
                    )

                    hidden_states = layer_outputs[0]
                    dist.send(hidden_states, dst=(rank + 1) % world_size)
                    past_key_values = layer_outputs[1]
                    new_keys = layer_outputs[2]
                    new_values = layer_outputs[3]
                    multi_decoder.post_forward(past_key_values, new_keys,
                                               new_values, cache_position)

                elif model_type == "qwen2":
                    past_seen_tokens = past_key_values.get_seq_length()
                    attention_mask = torch.ones([1, past_seen_tokens + 1], dtype=torch.int64)
                    position_ids = torch.arange(
                        past_seen_tokens,
                        1 + past_seen_tokens,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                    position_ids = position_ids.unsqueeze(0).view(-1, 1)

                    from transformers.modeling_attn_mask_utils \
                        import _prepare_4d_causal_attention_mask

                    causal_mask = _prepare_4d_causal_attention_mask(
                        attention_mask,
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        hidden_states,
                        past_seen_tokens,
                        sliding_window=model.model.config.sliding_window,
                    )
                    pad_len = multi_decoder.max_seq_len + 1 - causal_mask.size(-1)

                    causal_mask[:, :, :, -1] = torch.finfo(torch.float16).min
                    pad_mask = (0, pad_len)
                    padded_causal_mask = F.pad(
                        causal_mask.to(torch.float16), pad_mask,
                        value=torch.finfo(torch.float16).min
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


class BaseDecodeRunner:
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

    model_type = model.config.model_type
    print("MODEL TYPE: ", model_type)

    layer_start = 0
    layer_end = len(model.model.layers)
    num_heads = model.model.layers[layer_start].self_attn.num_heads
    num_key_value_heads = model.model.layers[layer_start].self_attn.num_key_value_heads
    head_dim = model.model.layers[layer_start].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    intermediate_size = model.config.intermediate_size
    deocderlayers = []
    layer_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    layer_indexs = range(layer_start, layer_end)
    for layer_idx in layer_indexs:
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        if model_type == "qwen2" and model.config.intermediate_size == 8960:
            # for qwen2-1.5b
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale),
            ]
        elif model_type == "qwen2" and model.config.intermediate_size == 18944:
            # for qwen2-7b
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj_0.weight, mlp_layer.down_proj_0.scale),
                (mlp_layer.down_proj_1.weight, mlp_layer.down_proj_1.scale)
            ]
        elif model_type == "llama":
            weights = [
                (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
                (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
                (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
                (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
                (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
                (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
                (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale),
            ]

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)

        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        if model_type == "llama":
            print("You are using llama")
            new_decoderlayer = llama.FusedLlamaLowBitDecoderlayer(
                weights,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                cached_cos=cached_cos,
                cached_sin=cached_sin,
                layer_norm_0=layer_norm_0,
                layer_norm_1=layer_norm_1,
                layer_idx=layer_idx,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                max_seq_len=max_output_len,
                transpose_value=transpose_value_cache,
            )
        elif model_type == "qwen2":
            print("You are using qwen")
            new_decoderlayer = qwen.FusedQwenLowBitDecoderlayer(
                weights,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                cached_cos=cached_cos,
                cached_sin=cached_sin,
                layer_norm_0=layer_norm_0,
                layer_norm_1=layer_norm_1,
                q_bias=attn_layer.q_proj.bias.to(torch.float16),
                k_bias=attn_layer.k_proj.bias.to(torch.float16),
                v_bias=attn_layer.v_proj.bias.to(torch.float16),
                layer_idx=layer_idx,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                max_seq_len=max_output_len,
                transpose_value=transpose_value_cache,
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

        if model_type == "llama":
            hidden_states, position_ids, causal_mask, past_key_values, cache_position = result
            with torch.inference_mode():
                for decoder_layer in deocderlayers:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=False,
                        use_cache=True,
                        cache_position=cache_position,
                    )

                    hidden_states = layer_outputs[0]
                    next_decoder_cache = layer_outputs[1]

                result_queue.put((hidden_states, next_decoder_cache))

        elif model_type == "qwen2":
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
                    )

                    hidden_states = layer_outputs[0]
                    next_decoder_cache = layer_outputs[1]
                result_queue.put((hidden_states, next_decoder_cache))


class BasePrefillRunner:
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

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()
