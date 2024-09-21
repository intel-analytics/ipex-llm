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
