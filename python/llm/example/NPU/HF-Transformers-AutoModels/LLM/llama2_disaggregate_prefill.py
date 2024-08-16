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
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["IPEX_LLM_LAST_LM_HEAD"] = "1"
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


@torch.no_grad()
def run_model(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: str,
    replica: int = 1,
) -> torch.Tensor:
    """Run a factory operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (Union[torch.Tensor, List[torch.Tensor]]): Activation tensor(s). Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        backend_cls (Any): Backend class to run
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Returns:
        torch.Tensor: result
    """
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


class LowBitLlamaMultiDecoderlayer(NNFactory):
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
        super().__init__(profile, device)
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.cached_cos = cached_cos
        self.cached_sin = cached_sin
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value

        cos = self.constant(self.cached_cos)
        self.cos = self.unsqueeze(cos, axis=0)

        sin = self.constant(self.cached_sin)
        self.sin = self.unsqueeze(sin, axis=0)
        
        if mode == "decode":
            assert self.seq_len == 1, "seq_len must be 1 for decode mode"
            self.kv_seq_len = self.max_seq_len + 1
        else:
            self.kv_seq_len = self.seq_len

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # define input, the order self.parameter matters
        input = self.parameter((self.batch_size, self.seq_len, self.hidden_size))

        # Self Attention
        if mode == "decode":
            attention_mask = self.parameter((self.batch_size, 1, 1, self.max_seq_len + 1))
        else:
            attention_mask = self.parameter((self.batch_size, 1, self.seq_len, self.seq_len))
            
        
        position_ids = self.parameter((self.batch_size, self.seq_len))
        past_keys = []
        past_values = []
        if mode == "decode":
            for i in range(num_layers):
                past_key = self.parameter((self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim))
                if transpose_value:
                    past_value = self.parameter((self.batch_size, self.num_key_value_heads, self.head_dim, self.max_seq_len))
                else:
                    past_value = self.parameter((self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim))
                past_keys.append(past_key)
                past_values.append(past_value)
        else:
            past_keys = [None] * num_layers
            past_values = [None] * num_layers
        
        if input_layernorm_weights is None:
            assert post_attn_layernorm_weights is None
            input_layernorm_weights = []
            post_attn_layernorm_weights = []
            for i in range(num_layers):
                input_layernorm_weights.append(self.parameter((1, self.hidden_size,)))
                post_attn_layernorm_weights.append(self.parameter((1, self.hidden_size,)))
        else:
            input_layernorm_weights = [self.constant(w) for w in input_layernorm_weights]
            post_attn_layernorm_weights = [self.constant(w) for w in post_attn_layernorm_weights]

        hidden_states = input
        
        curr_key_values = []
        for i in range(num_layers):
            hidden_states, new_key_states, new_value_states = self.build_decoder(hidden_states=hidden_states,
                                                                                 attention_mask=attention_mask,
                                                                                 position_ids=position_ids,
                                                                                 input_layernorm_weight=input_layernorm_weights[i],
                                                                                 post_attention_layernorm_weight=post_attn_layernorm_weights[i],
                                                                                 past_key=past_keys[i],
                                                                                 past_value=past_values[i],)
            curr_key_values.append((new_key_states, new_value_states))

        
        # define outputs
        hidden_states = self.convert_to_fp16(hidden_states)
        
        for i in range(num_layers):
            new_key_states = self.convert_to_fp16(curr_key_values[i][0])
            new_value_states = self.convert_to_fp16(curr_key_values[i][1])

        with FileLock("decoder_compile.lock"):
            print("start compiling")
            self.compile()

    def repeat_kv(self, hidden_states, n_rep, transpose=False):
        if n_rep == 1:
            return hidden_states
        if not transpose:
            hidden_states = self.reshape(hidden_states, [self.batch_size, self.num_key_value_heads, 1, self.kv_seq_len, self.head_dim])
            hidden_states = self.broadcast(hidden_states, [self.batch_size, self.num_key_value_heads, n_rep, self.kv_seq_len, self.head_dim])
            hidden_states = self.reshape(hidden_states, [self.batch_size, n_rep*self.num_key_value_heads, self.kv_seq_len, self.head_dim])
        else:
            hidden_states = self.reshape(hidden_states, [self.batch_size, self.num_key_value_heads, 1, self.head_dim, self.kv_seq_len])
            hidden_states = self.broadcast(hidden_states, [self.batch_size, self.num_key_value_heads, n_rep, self.head_dim, self.kv_seq_len])
            hidden_states = self.reshape(hidden_states, [self.batch_size, n_rep*self.num_key_value_heads, self.head_dim, self.kv_seq_len])
        return hidden_states

    def build_decoder(self, hidden_states, attention_mask, position_ids,
                      input_layernorm_weight, post_attention_layernorm_weight,
                      past_key = None,
                      past_value = None):

        residual = hidden_states

        input_2d = self.reshape(hidden_states, (self.batch_size * self.seq_len, self.hidden_size))

        # input layernorm
        input_2d = self.convert_to_fp32(input_2d)
        variance = self.reduce_mean(self.power(input_2d, self.constant(np.array([[2]], dtype=np.float32))), -1, keep_dims=True)
        eps = self.constant(self.rms_norm_eps)
        input_2d = self.eltwise_div(input_2d, self.sqrt(self.eltwise_add(variance, eps)))
        input_layernorm_weight = self.convert_to_fp32(input_layernorm_weight)
        input_2d = self.eltwise_mul(input_layernorm_weight, input_2d)
        input_2d = self.convert_to_fp16(input_2d)

        # attention
        query_states = self.linear(input_2d, self.num_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)
        key_states = self.linear(input_2d, self.num_key_value_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)
        value_states = self.linear(input_2d, self.num_key_value_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)

        query_states = self.reshape(query_states, [self.batch_size, self.seq_len, self.num_heads, self.head_dim])
        key_states = self.reshape(key_states, [self.batch_size, self.seq_len, self.num_key_value_heads, self.head_dim])
        value_states = self.reshape(value_states, [self.batch_size, self.seq_len, self.num_key_value_heads, self.head_dim])
        
        query_states = self.transpose(query_states, [0, 2, 1, 3])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        if self.transpose_value:
            value_states = self.transpose(value_states, [0, 2, 3, 1])
        else:
            value_states = self.transpose(value_states, [0, 2, 1, 3])
        
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, self.cos, self.sin, position_ids)
        new_key_states = key_states
        new_value_states = value_states
        
 
        
        if self.mode == "decode":
            key_states = self.concat(past_key, key_states, axis=-2)
            if self.transpose_value:
                value_states = self.concat(past_value, value_states, axis=-1)
            else:
                value_states = self.concat(past_value, value_states, axis=-2)
        
        # repeat_kv cannot be implemented because Broadcast op is needed
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups, self.transpose_value)
        
        attn_weight = self.matmul(query_states, key_states, False, True) / (math.sqrt(self.head_dim))
        attn_weight = self.eltwise_add(attn_weight, attention_mask)
        attn_weight = self.convert_to_fp32(attn_weight)
        attn_weight = self.softmax(attn_weight, -1)
        attn_weight = self.convert_to_fp16(attn_weight)
        attn_output = self.matmul(attn_weight, value_states, False, self.transpose_value)
        

        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(attn_output, [self.batch_size, self.seq_len, self.hidden_size])
        
        attn_output = self.linear(attn_output, self.hidden_size, self.hidden_size, bias=False, wt_dtype=self.dtype)

        hidden_states = self.eltwise_add(residual, attn_output)

        # Fully Connected
        residual = hidden_states
        # post_attention_layernorm forward
        
        hidden_states = self.convert_to_fp32(hidden_states)
        variance = self.reduce_mean(self.power(hidden_states, self.constant(np.array([[[2]]], dtype=np.float32))), -1, keep_dims=True)
        hidden_states = self.eltwise_div(hidden_states, self.sqrt(self.eltwise_add(variance, eps)))
        # post_attention_layernorm_weight = self.constant(post_attention_layernorm_weight)
        post_attention_layernorm_weight = self.convert_to_fp32(post_attention_layernorm_weight)
        hidden_states = self.eltwise_mul(post_attention_layernorm_weight, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)

        # mlp
        mm1 = self.linear(hidden_states, self.intermediate_size, self.hidden_size,
                          bias=False, wt_dtype=self.dtype)
        mm2 = self.linear(hidden_states, self.intermediate_size, self.hidden_size,
                          bias=False, wt_dtype=self.dtype)  # type: ignore[attr-defined]
        mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]

        hidden_states = self.linear(mm1, self.hidden_size, self.intermediate_size, bias=False, wt_dtype=self.dtype)
        
        hidden_states = self.eltwise_add(residual, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)

        return hidden_states, new_key_states, new_value_states
    
    def rotate_half(self, x):
        x1 = self.slice(x, [0, 0, 0, 0], [self.batch_size, self.num_heads, self.seq_len, self.head_dim//2], )
        x2 = self.slice(x, [0, 0, 0, self.head_dim//2], [self.batch_size, self.num_heads, self.seq_len, self.head_dim])
        return self.concat(self.negative(x2), x1, axis=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        position_ids = self.squeeze(position_ids)
        cos = self.gather(cos, self.convert_to_int32(position_ids), self.constant(1), 0)
        sin = self.gather(sin, self.convert_to_int32(position_ids), self.constant(1), 0)
        cos = self.unsqueeze(cos, [1])
        sin = self.unsqueeze(sin, [1])
        
        q_embed = self.eltwise_add(self.eltwise_mul(q, cos), self.eltwise_mul(self.rotate_half(q), sin))
        k_embed = self.eltwise_add(self.eltwise_mul(k, cos), self.eltwise_mul(self.rotate_half(k), sin))
        
        return q_embed, k_embed


class FusedLlamaLowBitMultiDecoderlayer(torch.nn.Module):

    def __init__(
        self,
        parameters: List[Tuple[torch.Tensor]],
        input_laynorm_weights: List[torch.Tensor],
        post_attn_layernorm_weights: List[torch.Tensor],
        layer_indexes : List[int],
        intra_stages: int,
        cached_cos,
        cached_sin,
        # rotary_emb,
        # batch_size: int,
        # seq_len: int,
        # hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int,
        rms_norm_eps,
        intermediate_size,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        do_print: bool = False
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
        # self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        # self.rotary_emb = rotary_emb
        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
            assert np_dtype == np.uint8
            assert parameters[0][1].dtype == torch.float16, parameters[0]
        else:  # FP16 Linear
            assert False, "should not be here"
            np_dtype = np.float16
        
        self.intra_stages = intra_stages
        self.layer_indexes = layer_indexes
        self.num_layers_1 = len(self.layer_indexes) // 2
        self.num_layers_0 = len(self.layer_indexes) - self.num_layers_1
        num_layers = len(self.layer_indexes) // intra_stages
        self.layer_ranges = []
        for i in range(intra_stages):
            if i == intra_stages - 1:
                self.layer_ranges.append((i * num_layers, len(self.layer_indexes)))
            else:
                self.layer_ranges.append((i * num_layers, (i+1) * num_layers))
        # last_num_layers = len(self.layer_indexes) - num_layers * (intra_stages -1)
        # self.stage_num_layers = [num_layers] * intra_stages + [last_num_layers]


        # assert self.num_layers_1 + self.num_layers_0 == len(input_laynorm_weights)
        # assert self.num_layers_1 + self.num_layers_0 == len(post_attn_layernorm_weights)

        self.backend_decoders = []

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            print(f"creating dedcoder layer: {self.layer_indexes[start]}:{self.layer_indexes[end - 1]}")
            decoder = LowBitLlamaMultiDecoderlayer([1, 1, num_heads*head_dim],
                                            input_layernorm_weights=input_laynorm_weights[start:end],
                                            post_attn_layernorm_weights=post_attn_layernorm_weights[start:end],
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
                                            dtype=np_dtype)
            self.backend_decoders.append(decoder)
            print(f"created dedcoder layer: {self.layer_indexes[start]}:{self.layer_indexes[end - 1]}")

        
        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            num_intra_layers = end - start
            self.backend_decoders[i].setWeights(3+(num_intra_layers)*2, self.op_id, *op_parameters[start*7:end*7])
            backend_lib.run(self.backend_decoders[i]._mm)

            print(f"{i}th inference done")

        self.kv_cache_c_parameter_handel = []
        self.kv_cache_parameters = []
        self.kv_cache_prefetched = False


    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        
        inputs = (hidden_states.to(torch.float16),
                  attention_mask,
                  position_ids,
                )

        if len(self.kv_cache_parameters) > 0:
            # the case kv cache changed
            cached_prt = self.kv_cache_parameters[0].storage().data_ptr()
            current_ptr = past_key_value.key_cache[self.layer_indexes[0]].storage().data_ptr()
            if cached_prt != current_ptr:
                self.kv_cache_parameters = []
                self.kv_cache_c_parameter_handel = []
                self.kv_cache_prefetched = False

        if len(self.kv_cache_parameters) == 0:
            for idx in self.layer_indexes:
                past_key = past_key_value.key_cache[idx]
                past_value = past_key_value.value_cache[idx]

                assert past_key.dtype == torch.float16, f"past_key dtype is {past_key.dtype}"

                new_size = (past_key.size(0),
                            past_key.size(1),
                            self.max_seq_len,
                            past_key.size(3))
                past_key = past_key.as_strided(new_size, past_key.stride(), storage_offset=0)
                assert past_key.is_contiguous()
                past_value = past_value.as_strided(new_size, past_value.stride(), storage_offset=0)
                if self.transpose_value:
                    past_value = past_value.transpose(-1, -2)
                assert past_value.is_contiguous()

                self.kv_cache_parameters.append(past_key)
                self.kv_cache_parameters.append(past_value)
            
            for i in range(self.intra_stages):
                start, end = self.layer_ranges[i]
                handle = self.backend_decoders[i].create_parameters([p.numpy() for p in self.kv_cache_parameters[start*2:end*2]])
                self.kv_cache_c_parameter_handel.append(handle)

        x_np = [elem.to(torch.float16).numpy() for elem in inputs]

        key_value_states = []

        with record_function(f"npu_factory"):
            if not self.kv_cache_prefetched:
                for i in range(self.intra_stages):
                    self.backend_decoders[i].load_wt_fn(len(inputs), self.backend_decoders[i]._mm, self.kv_cache_c_parameter_handel[i])

            models_ptr = (ctypes.POINTER(ctypes.c_char) * self.intra_stages)(*[self.backend_decoders[i]._mm for i in range(self.intra_stages)])
            inputs_ptr = (ctypes.c_void_p * 3)(x_np[0].ctypes.data_as(ctypes.c_void_p), x_np[1].ctypes.data_as(ctypes.c_void_p), x_np[2].ctypes.data_as(ctypes.c_void_p))
            t0 = time.perf_counter()
            backend_lib.run_decoders(models_ptr, inputs_ptr, 2, 3)
            t1 = time.perf_counter()

        hidden_states = self.backend_decoders[-1].torch_out[0]

        if self.do_print:
            print("outputs:", hidden_states)

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs, t1 - t0
    
    def post_forward(self, past_key_value, cache_position):
        key_value_states = []
        for i in range(self.intra_stages):
            for j in range(1, len(self.backend_decoders[i].torch_out)):
                key_value_states.append(self.backend_decoders[i].torch_out[j])
        
        cache_kwargs = {"cache_position": cache_position, "max_seq_len":self.max_seq_len, "transpose": self.transpose_value}
        for i in range(len(self.layer_indexes)):
            key_states, value_states = past_key_value.update(key_value_states[2*i],
                                                             key_value_states[2*i+1],
                                                             self.layer_indexes[i], cache_kwargs)
        
        for i in range(self.intra_stages):
            self.backend_decoders[i].load_wt_fn(3, self.backend_decoders[i]._mm, self.kv_cache_c_parameter_handel[i])
        self.kv_cache_prefetched = True


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

        self.backend_cls_prefill = partial(LowBitLlamaMultiDecoderlayer,
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
                                           dtype=np_dtype)
        self.layer_norm_0 = layer_norm_0
        self.layer_norm_1 = layer_norm_1

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        assert not output_attentions

        seq_len = hidden_states.shape[1]
        assert seq_len > 1, "seq_len must be 1 for decode mode"

        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16), attention_mask, position_ids)
        inputs += (self.layer_norm_0, self.layer_norm_1)
        hidden_states, past_key, past_value = run_model(inputs, self.op_parameters, backend_cls, self.op_id, replica=2)
        cache_kwargs = {"cache_position": cache_position, "max_seq_len":self.max_seq_len, "transpose": self.transpose_value}
        key_states, value_states = past_key_value.update(past_key, past_value, self.layer_idx, cache_kwargs)

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs

import time
import types
def run_decode(model, rank, world_size, port, layer_start, layer_end,
               intra_stages, max_seq_len, transpose_value_cache,
               input_queue, result_queue):
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

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
    deocderlayers = []
    layer_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    layer_indexs = range(layer_start, layer_end)
    for layer_idx in layer_indexs:
        curr_layer = model.model.layers[layer_idx]
        attn_layer = curr_layer.self_attn
        mlp_layer = curr_layer.mlp

        weights = [
            (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
            (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
            (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
            (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
            (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
            (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
            (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale)]

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)

        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)
        
        layer_weights.extend(weights)
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)
    
    multi_decoder = FusedLlamaLowBitMultiDecoderlayer(
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
        do_print=False, # layer_start == 0,
    )

    result_queue.put("loading success")

    past_key_values = None

    control = torch.empty((), dtype=torch.int)
    hidden_states = torch.empty((1, 1, head_dim*num_heads), dtype=torch.float16)
    with torch.inference_mode():
        while True:
            
            dist.broadcast(control, src=0)
            if control.item() == -2:
                break
            elif control.item() == -1:
                past_key_values = input_queue.get()
            else:
                t0 = time.perf_counter()
                past_seen_tokens = past_key_values.get_seq_length()
                attention_mask = torch.ones([1, past_seen_tokens + 1], dtype=torch.int64)
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1,
                                              device=hidden_states.device)

                position_ids = position_ids = cache_position.unsqueeze(0)
                causal_mask = model.model._update_causal_mask(attention_mask, hidden_states,
                                                    cache_position, past_seen_tokens)
                pad_len = multi_decoder.max_seq_len + 1 - causal_mask.size(-1)

                pad_mask = (0, pad_len)
                padded_causal_mask = F.pad(causal_mask.to(torch.float16), pad_mask,
                                        value=torch.finfo(torch.float16).min)
                padded_causal_mask[:,:,:,-1] = 0.0
                dist.recv(hidden_states, src=rank - 1)
                t1 = time.perf_counter()
                layer_outputs, elapse = multi_decoder(hidden_states,
                        attention_mask=padded_causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=False,
                        use_cache=True,
                        cache_position=cache_position,)
                t2 = time.perf_counter()
                hidden_states = layer_outputs[0]
                t3 = time.perf_counter()
                dist.send(hidden_states, dst=(rank + 1)%world_size)
                t4 = time.perf_counter()
                multi_decoder.post_forward(past_key_values, cache_position)

import time
class DecodeRunner:
    def __init__(self, model, max_seq_len, transpose_value_cache):
        self.model = model
        self.max_seq_len = max_seq_len
        self.transpose_value_cache = transpose_value_cache
        world_size = 3
        num_layers = self.model.model.config.num_hidden_layers

        port = '54791'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(world_size)

        self.input_queues = []
        self.output_queues = []
        self.decoder_processes = []

        for rank in range(1, world_size):
            input_q = mp.Queue()
            output_q = mp.Queue()
            start_layer = (rank - 1) * (num_layers // (world_size - 1))
            end_layer = (rank) * (num_layers // (world_size - 1))
            if rank == world_size - 1:
                end_layer = num_layers
            p = mp.Process(target=run_decode, args=(self.model,
                                                    rank, world_size, port,
                                                    start_layer, end_layer,
                                                    2,
                                                    self.max_seq_len,
                                                    self.transpose_value_cache,
                                                    input_q,
                                                    output_q))
            p.daemon = True
            p.start()
            self.input_queues.append(input_q)
            self.output_queues.append(output_q)
            self.decoder_processes.append(p)

        dist.init_process_group()
        my_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        logger.info(f"rank: {my_rank}, size: {self.world_size}")

        for i, p in enumerate(self.output_queues):
            output = self.output_queues[i].get()
            print(Fore.GREEN  + f"decode process {i + 1} output: {output}")
            print(Style.RESET_ALL)

        self.cache_past_key_value = None

    def forward(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,):
            t0 = time.perf_counter()

            if self.cache_past_key_value != past_key_value:
                control = torch.tensor(-1, dtype=torch.int)
                dist.broadcast(control, src=0)
                for i in range(len(self.decoder_processes)):
                    self.input_queues[i].put(past_key_value)

            control = torch.tensor(0, dtype=torch.int)
            dist.broadcast(control, src=0)
            hidden_states = hidden_states.to(torch.float16)
            dist.send(hidden_states, dst=1)
            past_key_value.expand()
            dist.recv(hidden_states, src=self.world_size - 1)
            t1 = time.perf_counter()
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

def run_prefill(model, max_seq_len, transpose_value_cache, input_queue, result_queue):
    print("finish loading prefill model")

    # from ipex_llm.transformers.npu_models.convert import optimize_llm, optimize_llm_post

    # optimize_llm(model)
    
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

        weights = [
            (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
            (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
            (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
            (attn_layer.o_proj.weight, attn_layer.o_proj.scale),
            (mlp_layer.gate_proj.weight, mlp_layer.gate_proj.scale),
            (mlp_layer.up_proj.weight, mlp_layer.up_proj.scale),
            (mlp_layer.down_proj.weight, mlp_layer.down_proj.scale)]

        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)

        layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
        layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

        new_decoderlayer = FusedLlamaLowBitDecoderlayer(weights,
                                            num_heads=num_heads,
                                            num_key_value_heads=num_key_value_heads,
                                            cached_cos=cached_cos,
                                            cached_sin=cached_sin,
                                            layer_norm_0=layer_norm_0,
                                            layer_norm_1=layer_norm_1,
                                            layer_idx=layer_idx,
                                            rms_norm_eps=rms_norm_eps,
                                            intermediate_size=intermediate_size,
                                            max_seq_len=max_seq_len,
                                            transpose_value=transpose_value_cache)
        
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


class PrefillRunner:
    def __init__(self, model, max_seq_len, transpose_value_cache):
        self.model = model
        self.max_seq_len = max_seq_len
        self.transpose_value_cache = transpose_value_cache

        self.prefill_result_queue = mp.Queue()
        self.prefill_input_queue = mp.Queue()

        self.p = mp.Process(target=run_prefill, args=(model,
                                                      args.max_seq_len,
                                                      args.transpose_value_cache,
                                                      self.prefill_input_queue,
                                                      self.prefill_result_queue))
        self.p.daemon = True
        self.p.start()
        output = self.prefill_result_queue.get()
        print(Fore.GREEN + f"prefill process output: {output}")
        print(Style.RESET_ALL)

    def forward(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,):
            self.prefill_input_queue.put((hidden_states, position_ids, attention_mask, past_key_value, cache_position))
            return self.prefill_result_queue.get()
    
    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
def gen_llama_fused_model_forward(prefill_runner, decode_runner):
    def llama_fused_model_forward(
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
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        t0 = time.perf_counter()
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            invalidInputError(False,
                            ("You cannot specify both input_ids and inputs_embeds at the same time, "
                            "and must specify either one"))

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0

        # ipex-llm changes start
        from ipex_llm.transformers.npu_models.kv import DynamicFusedNormalCache
        if use_cache and not isinstance(past_key_values, DynamicFusedNormalCache):
            past_key_values = DynamicFusedNormalCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                                        device=inputs_embeds.device)
        # ipex-llm changes end

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                            cache_position, past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        seq_len = hidden_states.size(1)

        if seq_len == 1:
            layers_runner = decode_runner
        else:
            layers_runner = prefill_runner
        layer_outputs = layers_runner.forward(hidden_states,
                                              attention_mask=causal_mask,
                                              position_ids=position_ids,
                                              past_key_value=past_key_values,
                                              output_attentions=output_attentions,
                                              use_cache=use_cache,
                                              cache_position=cache_position,)
        hidden_states = layer_outputs[0]

        next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ipex-llm changes start
        next_cache = next_decoder_cache if use_cache else None
        # ipex-llm changes end
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache,
                                    all_hidden_states, all_self_attns] if v is not None)
        t1 = time.perf_counter()
        # print("fused model forward time: ", t1 - t0)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    return llama_fused_model_forward

def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)

from transformers.models.llama.modeling_llama import LlamaModel

import torch.multiprocessing as mp
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for npu model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="D:\llm-models\Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=64,
                        help='Max tokens to predict')
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--transpose-value-cache', action="store_true", default=False)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 trust_remote_code=True, attn_implementation="eager",
                                                 load_in_low_bit="sym_int4")

    model.share_memory()

    print("share memory success")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt_1024 = '''In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary.
In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined.
One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms.
In fact, it was hard to find anything in people's lives that wasn't touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve.
Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future.
Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone.
One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation.
Maya's work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts.
One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say.
As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility.
When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage.
As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives.
Overall, Maya's speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all. 
As Maya left the stage and made her way back to her seat, she couldn't help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work. 
For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone.
As she'''
    prompt = args.prompt
    prompt = prompt_1024
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    input_ids = input_ids[:, :args.max_seq_len - args.n_predict]


    decode_runner = DecodeRunner(model, args.max_seq_len, args.transpose_value_cache)
    prefill_runner = PrefillRunner(model, args.max_seq_len, args.transpose_value_cache)

    llama_model_forward = gen_llama_fused_model_forward(prefill_runner=prefill_runner,
                                  decode_runner=decode_runner)
    convert_forward(model, LlamaModel, llama_model_forward)

    from ipex_llm.utils.benchmark_util_4_29 import BenchmarkWrapper

    model = BenchmarkWrapper(model, do_print=True)

    with torch.inference_mode():
        # input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print("finish to load")
        print('input length:', len(input_ids[0]))
        for i in range(3):
            st = time.time()
            output = model.generate(input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict)
            end = time.time()
            print(f'Inference time: {end-st} s')

            print('-'*20, 'Input', '-'*20)
            input_str = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(input_str)

            output_str = tokenizer.decode(output[0], skip_special_tokens=False)

            print('-'*20, 'Output', '-'*20)
            print(output_str)

    print('-'*80)
    print('done')

    decode_runner.shutdown()
    prefill_runner.shutdown()

    print("success shut down")




