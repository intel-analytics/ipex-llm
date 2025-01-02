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
from typing import Optional, Any, List
import numpy.typing as npt
import os

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
    x_np = [set_contiguous(elem).numpy() for elem in x]
    op_args = []
    op_args_flatten = []
    for w in weights:
        if isinstance(w, tuple):  # from QuantizedLinear
            if len(w) == 2:
                op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy()))
                op_args_flatten.append(op_args[-1][0])
                op_args_flatten.append(op_args[-1][1])
            else:
                op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy(),
                                set_contiguous(w[2]).numpy()))
                op_args_flatten.append(op_args[-1][0])
                op_args_flatten.append(op_args[-1][1])
                op_args_flatten.append(op_args[-1][2])
        elif w.dtype in [torch.int8, torch.uint8]:    # QuantizedLinear weight
            op_args.append(w.numpy())
            op_args_flatten.append(op_args[-1])
        elif isinstance(w, np.ndarray):     # scale
            op_args.append(w)
            op_args_flatten.append(op_args[-1])
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

    def __init__(self, max_seq_len, transpose_value, dtype, profile=False, device="NPU",
                 n_splits_linear=1, n_splits_down_proj=1, group_size=0, asym=False):
        super().__init__(profile, device)
        self.cache_parameter_ops = []
        self.input_ops = []
        self.linear_ops = []
        self.kv_cache_c_handle = None
        self.kv_cache_torch = []
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        self.dtype = dtype
        self.n_splits_linear = n_splits_linear
        self.n_splits_down_proj = n_splits_down_proj
        self.group_size = group_size
        self.asym = asym

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
                  v_bias=None,
                  use_prefill_sdp=False):
        hidden_size = num_heads * head_dim
        num_key_value_groups = num_heads // num_key_value_heads
        if self.n_splits_linear != 1:
            hidden_states = self.unsqueeze(hidden_states, axis=0)

        query_states = self.linear(
            hidden_states,
            num_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )

        key_states = self.linear(
            hidden_states,
            num_key_value_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )

        value_states = self.linear(
            hidden_states,
            num_key_value_heads * head_dim,
            hidden_size,
            bias=False,
            wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )

        if q_bias is not None:
            query_states = query_states + q_bias
        if k_bias is not None:
            key_states = key_states + k_bias
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
        use_ov_sdp = (mode == "prefill") and use_prefill_sdp
        if self.transpose_value:
            new_value_states = self.transpose(value_states, [0, 2, 3, 1])
            if use_ov_sdp:
                value_states = self.transpose(value_states, [0, 2, 1, 3])
            else:
                value_states = new_value_states
        else:
            value_states = self.transpose(value_states, [0, 2, 1, 3])
            new_value_states = value_states

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
                                      transpose=(self.transpose_value and (not use_ov_sdp)))
        if use_ov_sdp:
            value_states = self.convert_to_fp32(value_states)
            key_states = self.convert_to_fp32(key_states)
            query_states = self.convert_to_fp32(query_states)
            attn_output = self.scaled_dot_product_attention(
                query_states, key_states, value_states, None, True)
            attn_output = self.convert_to_fp16(attn_output)
        else:
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
            attn_output, hidden_size, hidden_size, bias=False, wt_dtype=self.dtype,
            n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )
        return attn_output, new_key_states, new_value_states

    def paraformer_layer_norm(self, hidden_states, layernorm_weight, layernorm_bias):
        hidden_states = self.convert_to_fp32(hidden_states)
        mean_res = self.reduce_mean(hidden_states, -1, keep_dims=True,)
        variance = self.reduce_mean(
            self.power(hidden_states - mean_res, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(1e-12)
        hidden_states = self.eltwise_div(hidden_states - mean_res,
                                         self.sqrt(self.eltwise_add(variance, eps)))
        layernorm_weight = self.convert_to_fp32(layernorm_weight)
        hidden_states = self.eltwise_mul(layernorm_weight, hidden_states)
        layernorm_bias = self.convert_to_fp32(layernorm_bias)
        hidden_states = self.eltwise_add(layernorm_bias, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)
        return hidden_states

    def self_attn_sanm(self,
                       x,
                       mask,
                       in_feat,
                       n_feat,
                       n_head,
                       fsmn_weight,
                       qkv_bias,
                       out_bias
                       ):
        d_k = n_feat // n_head
        h = n_head
        b, t, d = x.size()

        q_k_v = self.linear(x,
                            1536,
                            512,
                            bias=False,
                            wt_dtype=self.dtype)
        q_k_v = self.eltwise_add(q_k_v, qkv_bias)

        q = self.slice(q_k_v, [0, 0, 0], [1, 218, 512])
        k = self.slice(q_k_v, [0, 0, 512], [1, 218, 2 * 512])
        v = self.slice(q_k_v, [0, 0, 2 * 512], [1, 218, 3 * 512])

        q_h = self.reshape(q, [b, t, h, d_k])
        k_h = self.reshape(k, [b, t, h, d_k])
        v_h = self.reshape(v, [b, t, h, d_k])
        q_h = self.transpose(q_h, [0, 2, 1, 3])
        k_h = self.transpose(k_h, [0, 2, 1, 3])
        v_h = self.transpose(v_h, [0, 2, 1, 3])

        b_v, t_v, d_v = v.size()
        if mask is not None:
            mask = self.reshape(mask, [b_v, -1, 1])
            v = self.eltwise_mul(v, mask)
        v_x = self.transpose(v, [0, 2, 1])

        fsmn_out = self.convolution(input_node=v_x,
                                    weights_node=fsmn_weight,
                                    bias=None,
                                    strides=1,
                                    padding=5,
                                    groups=512,
                                    n_spatial_dims=1)

        fsmn_out = self.transpose(fsmn_out, [0, 2, 1])
        fsmn_out = self.eltwise_add(v, fsmn_out)
        if mask is not None:
            fsmn_out = self.eltwise_mul(fsmn_out, mask)

        q_h = q_h * d_k ** (-0.5)
        scores = self.matmul(q_h, k_h, False, True)
        n_batch = v_h.size(0)
        p_attn = self.softmax(scores, -1)

        x_attn = self.matmul(p_attn, v_h, False, False)
        x_attn = self.transpose(x_attn, [0, 2, 1, 3])
        x_attn = self.reshape(x_attn, [n_batch, -1, h * d_k])

        attn_out = self.linear(x_attn,
                               512,
                               512,
                               bias=False,
                               wt_dtype=self.dtype)
        attn_out = attn_out + out_bias
        attn_res = self.eltwise_add(attn_out, fsmn_out)
        return attn_res

    def sanm_feed_forward(self, x, hidden_units, idim, w1_bias, w2_bias):
        mm = self.linear(x, 2048, 512, bias=False, wt_dtype=self.dtype)
        mm = mm + w1_bias
        mm_act = self.relu(mm)
        output = self.linear(mm_act, 512, 2048, bias=False, wt_dtype=self.dtype)
        output = output + w2_bias
        return output

    def multihead_attn_sanm_decoder(self, inputs, mask, fsmn_weight):
        b, t, d = inputs.size()
        if mask is not None:
            mask = self.reshape(mask, [b, -1, 1])
            inputs = self.eltwise_mul(inputs, mask)
        x = self.transpose(inputs, [0, 2, 1])
        b, d, t = x.size()
        cache = x
        fsmn_x = self.convolution(input_node=x,
                                  weights_node=fsmn_weight,
                                  bias=None,
                                  strides=1,
                                  padding=5,
                                  groups=512,
                                  n_spatial_dims=1)
        fsmn_x = self.transpose(fsmn_x, [0, 2, 1])
        x = self.eltwise_add(inputs, fsmn_x)
        if mask is not None:
            x = self.eltwise_mul(x, mask)
        return x, cache

    def sanm_cross_attn(self, x, memory, mask, q_bias, kv_bias, out_bias, n_feat, n_head):
        b = x.size(0)
        d_k = n_feat // n_head
        h = n_head

        q = self.linear(x, 512, 512, bias=False, wt_dtype=self.dtype)
        q = q + q_bias
        q_h = self.reshape(q, (b, -1, h, d_k))
        q_h = self.transpose(q_h, [0, 2, 1, 3])

        k_v = self.linear(memory, 1024, 512, bias=False, wt_dtype=self.dtype)
        k_v = k_v + kv_bias

        res = k_v.chunk(2, -1)
        k = res[0]
        v = res[1]
        k_h = self.reshape(k, [b, -1, h, d_k])
        v_h = self.reshape(v, [b, -1, h, d_k])
        k_h = self.transpose(k_h, [0, 2, 1, 3])
        v_h = self.transpose(v_h, [0, 2, 1, 3])

        q_h = q_h * d_k ** (-0.5)
        scores = self.matmul(q_h, k_h, False, True)
        n_batch = v_h.size(0)
        # Assume mask is None
        p_attn = self.softmax(scores, -1)

        v_h = self.transpose(v_h, [0, 1, 3, 2])
        x_attn = self.matmul(p_attn, v_h, False, True)
        x_attn = self.transpose(x_attn, [0, 2, 1, 3])
        x_attn = self.reshape(x_attn, [n_batch, -1, h * d_k])
        attn_out = self.linear(x_attn, 512, 512, bias=False, wt_dtype=self.dtype)
        attn_out = attn_out + out_bias
        return attn_out

    def feed_forward_sanm_decoder(self, x, w_1_bias, norm_weights, norm_bias):
        w_1 = self.linear(x, 2048, 512, bias=False, wt_dtype=self.dtype)
        w_1 = w_1 + w_1_bias
        w_1_act = self.relu(w_1)
        w_1_norm = self.paraformer_layer_norm(w_1_act, norm_weights, norm_bias)
        w_2 = self.linear(w_1_norm, 512, 2048, bias=False, wt_dtype=self.dtype)
        return w_2

    def mlp(self, hidden_states, seq_len=-1, mode="prefill"):
        mm1 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False,
            wt_dtype=self.dtype, n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )
        mm2 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False,
            wt_dtype=self.dtype, n_splits=self.n_splits_linear,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
        )  # type: ignore[attr-defined]
        mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]

        hidden_states = self.linear(
            mm1, self.hidden_size, self.intermediate_size, bias=False, wt_dtype=self.dtype,
            n_splits=self.n_splits_down_proj,
            scale_factor=(self.group_size == 0),
            is_prefill=(mode == "prefill"),
            asym=self.asym
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
        if os.environ.get("IPEX_LLM_NPU_DRIVER_VERSION", None) in ["5716", "5733"]:
            # to support special drivers
            hidden_states = self.convert_to_fp16(hidden_states)
        else:
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
        if position_ids is not None:
            if os.environ.get("IPEX_LLM_NPU_MTL", "0") == "1" or\
               os.environ.get("IPEX_LLM_NPU_ARL", "0") == "1":
                position_ids = self.reshape(position_ids, [-1])
            else:
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

    def create_input_op(self, shape, dtype=np.float16):
        invalidInputError(len(self.cache_parameter_ops) == 0,
                          "create_input_op should be called before any create_cache_op")
        invalidInputError(len(self.linear_ops) == 0,
                          "create_input_op should be called before any linear op")

        op = super().parameter(shape, dtype)
        self.input_ops.append(op)
        return op

    def linear(self,
               input_node: ctypes._Pointer,
               output_channels: int,
               input_channels: int,
               bias: Optional[bool] = False,
               act_dtype: npt.DTypeLike = np.float16,
               wt_dtype: npt.DTypeLike = np.float16,
               n_splits: int = 1,
               scale_factor: bool = True,
               is_prefill: bool = False,
               asym: bool = False):
        if n_splits == 1:
            op = super().linear(input_node, output_channels,
                                input_channels, bias, act_dtype,
                                wt_dtype, scale_factor=scale_factor,
                                asym=asym)
        else:
            op = super().dq_split_linear(input_node, n_splits,
                                         output_channels, input_channels,
                                         bias=bias, act_dtype=act_dtype,
                                         wt_dtype=wt_dtype, scale_factor=scale_factor,
                                         is_prefill=is_prefill,
                                         asym=asym)
        self.linear_ops.append(op)
        return op

    def dq_split_linear(self,
                        input_node: ctypes._Pointer,
                        output_channels: int,
                        input_channels: int,
                        n_splits: int,
                        act_dtype: npt.DTypeLike = np.float16,
                        wt_dtype: npt.DTypeLike = np.float16,
                        scale_factor: bool = False,
                        is_prefill: bool = False,
                        asym: bool = False):
        op = super().dq_split_linear(input_node, n_splits, output_channels, input_channels,
                                     False, act_dtype, wt_dtype, scale_factor,
                                     is_prefill=is_prefill, asym=asym)
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
        x_np = [elem.numpy() for elem in inputs]

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
