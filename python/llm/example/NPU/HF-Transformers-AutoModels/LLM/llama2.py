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
        # variance = self.reduce_mean(self.eltwise_mul(input_2d, input_2d), -1, keep_dims=True)
        variance = self.reduce_mean(self.power(input_2d, self.constant(np.array([[2]], dtype=np.float32))), -1, keep_dims=True)
        eps = self.constant(self.rms_norm_eps)
        input_2d = self.eltwise_div(input_2d, self.sqrt(self.eltwise_add(variance, eps)))
        # input_layernorm_weight = self.constant(input_layernorm_weight)
        input_layernorm_weight = self.convert_to_fp32(input_layernorm_weight)
        input_2d = self.eltwise_mul(input_layernorm_weight, input_2d)
        input_2d = self.convert_to_fp16(input_2d)

        # attention
        query_states = self.linear(input_2d, self.num_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)
        key_states = self.linear(input_2d, self.num_key_value_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)
        value_states = self.linear(input_2d, self.num_key_value_heads*self.head_dim, self.hidden_size, bias=False, wt_dtype=self.dtype)
        
        # cos = self.constant(self.cached_cos)
        # cos = self.unsqueeze(cos, axis=0)

        # sin = self.constant(self.cached_sin)
        # sin = self.unsqueeze(sin, axis=0)

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
    ):
        super().__init__()

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
        
        self.layer_indexes = layer_indexes
        self.num_layers_1 = len(self.layer_indexes) // 2
        self.num_layers_0 = len(self.layer_indexes) - self.num_layers_1

        assert self.num_layers_1 + self.num_layers_0 == len(input_laynorm_weights)
        assert self.num_layers_1 + self.num_layers_0 == len(post_attn_layernorm_weights)

        print("create dedcoder layer")
        self.backend_cls_decode_0 = LowBitLlamaMultiDecoderlayer([1, 1, num_heads*head_dim],
                                          input_layernorm_weights=input_laynorm_weights[:self.num_layers_0],
                                          post_attn_layernorm_weights=post_attn_layernorm_weights[:self.num_layers_0],
                                          cached_cos=cached_cos,
                                          cached_sin=cached_sin,
                                          num_heads=num_heads,
                                          num_key_value_heads=num_key_value_heads,
                                          num_layers=self.num_layers_0,
                                          max_seq_len=max_seq_len,
                                          rms_norm_eps=rms_norm_eps,
                                          intermediate_size=intermediate_size,
                                          mode="decode",
                                          transpose_value=self.transpose_value,
                                          dtype=np_dtype)
        self.backend_cls_decode_1 = LowBitLlamaMultiDecoderlayer([1, 1, num_heads*head_dim],
                                          input_layernorm_weights=input_laynorm_weights[self.num_layers_0:],
                                          post_attn_layernorm_weights=post_attn_layernorm_weights[self.num_layers_0:],
                                          cached_cos=cached_cos,
                                          cached_sin=cached_sin,
                                          num_heads=num_heads,
                                          num_key_value_heads=num_key_value_heads,
                                          num_layers=self.num_layers_1,
                                          max_seq_len=max_seq_len,
                                          rms_norm_eps=rms_norm_eps,
                                          intermediate_size=intermediate_size,
                                          mode="decode",
                                          transpose_value=self.transpose_value,
                                          dtype=np_dtype)
        print("created dedcoder layer")

        assert (self.num_layers_0 + self.num_layers_1) * 7 == len(op_parameters)
        
        self.backend_cls_decode_0.setWeights(3+self.num_layers_0*2, self.op_id, *op_parameters[:self.num_layers_0*7])
        backend_lib.run(self.backend_cls_decode_0._mm)

        print("first inference done")

        self.backend_cls_decode_1.setWeights(3+self.num_layers_1*2, self.op_id, *op_parameters[self.num_layers_0*7:])


        print("weight setted")
        backend_lib.run(self.backend_cls_decode_1._mm)

        print("2nd inference done")

        self.kv_cache_c_parameter_handel = (None, None)
        self.kv_cache_parameters = None
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
        seq_len = hidden_states.shape[1]
        pad_len = self.max_seq_len + 1 - attention_mask.size(-1)

        pad_mask = (0, pad_len)
        padded_attention_mask = F.pad(attention_mask.to(torch.float16), pad_mask,
                                value=torch.finfo(torch.float16).min)
        padded_attention_mask[:,:,:,-1] = 0.0
        inputs = (hidden_states.to(torch.float16),
                    padded_attention_mask,
                    position_ids,
                    )

        if self.kv_cache_parameters is None:
            self.kv_cache_parameters = []
            self.kv_cache_c_parameter_handel = None
            self.kv_cache_prefetched = False
        else:
            # the case kv cache changed
            cached_prt = self.kv_cache_parameters[0].storage().data_ptr()
            current_ptr = past_key_value.key_cache[self.layer_indexes[0]].storage().data_ptr()
            if cached_prt != current_ptr:
                # print("kv cache changed")
                self.kv_cache_parameters = []
                self.kv_cache_c_parameter_handel = (None, None)
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
            handle_0 = self.backend_cls_decode_0.create_parameters([p.numpy() for p in self.kv_cache_parameters[:self.num_layers_0*2]])
            handle_1 = self.backend_cls_decode_1.create_parameters([p.numpy() for p in self.kv_cache_parameters[self.num_layers_0*2:]])
            assert len(self.kv_cache_parameters) == (self.num_layers_0 + self.num_layers_1) * 2
            self.kv_cache_c_parameter_handel = (handle_0, handle_1)

        x_np = [elem.to(torch.float16).numpy() for elem in inputs]

        key_value_states = []

        with record_function(f"npu_factory"):
            if not self.kv_cache_prefetched:
                self.backend_cls_decode_0.load_wt_fn(len(inputs), self.backend_cls_decode_0._mm, self.kv_cache_c_parameter_handel[0])
                self.backend_cls_decode_1.load_wt_fn(len(inputs), self.backend_cls_decode_1._mm, self.kv_cache_c_parameter_handel[1])

            models_ptr = (ctypes.POINTER(ctypes.c_char) * 2)(self.backend_cls_decode_0._mm, self.backend_cls_decode_1._mm)
            inputs_ptr = (ctypes.c_void_p * 3)(x_np[0].ctypes.data_as(ctypes.c_void_p), x_np[1].ctypes.data_as(ctypes.c_void_p), x_np[2].ctypes.data_as(ctypes.c_void_p))

            backend_lib.run_decoders(models_ptr, inputs_ptr, 2, 3)

        for i in range(1, len(self.backend_cls_decode_0.torch_out)):
            key_value_states.append(self.backend_cls_decode_0.torch_out[i])
        
        for i in range(1, len(self.backend_cls_decode_1.torch_out)):
            key_value_states.append(self.backend_cls_decode_1.torch_out[i])

        hidden_states = self.backend_cls_decode_1.torch_out[0]
        
        cache_kwargs = {"cache_position": cache_position, "max_seq_len":self.max_seq_len, "transpose": self.transpose_value}
        for i in range(len(self.layer_indexes)):
            key_states, value_states = past_key_value.update(key_value_states[2*i],
                                                             key_value_states[2*i+1],
                                                             self.layer_indexes[i], cache_kwargs)
        
        self.backend_cls_decode_0.load_wt_fn(len(inputs), self.backend_cls_decode_0._mm, self.kv_cache_c_parameter_handel[0])
        self.backend_cls_decode_1.load_wt_fn(len(inputs), self.backend_cls_decode_1._mm, self.kv_cache_c_parameter_handel[1])
        self.kv_cache_prefetched = True

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs


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
        # assert cache_position is None
        # assert use_cache

        seq_len = hidden_states.shape[1]

        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16), attention_mask, position_ids)
        inputs += (self.layer_norm_0, self.layer_norm_1)
        # print("start run_model prefill")
        hidden_states, past_key, past_value = run_model(inputs, self.op_parameters, backend_cls, self.op_id, replica=1)
        # print("end run model prefill")
        cache_kwargs = {"cache_position": cache_position, "max_seq_len":self.max_seq_len, "transpose": self.transpose_value}
        key_states, value_states = past_key_value.update(past_key, past_value, self.layer_idx, cache_kwargs)

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for npu model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--transpose-value-cache', action="store_true", default=False)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    dist.init_process_group()
    my_rank = dist.get_rank()
    my_size = dist.get_world_size()
    logger.info(f"rank: {my_rank}, size: {my_size}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 trust_remote_code=True, attn_implementation="eager",
                                                 load_in_low_bit="sym_int4", pipeline_parallel_stages=2)

    if my_rank == 0:
        print(model)
    dist.barrier()

    if my_rank == 1:
        print(model)

    layer_start = model.layer_start
    layer_end = model.layer_end
    num_layers = model.num_layers
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
                                            max_seq_len=args.max_seq_len,
                                            transpose_value=args.transpose_value_cache)
        
        layer_weights.extend(weights)
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)
        model.model.layers[layer_idx] = new_decoderlayer

    multi_decoder = FusedLlamaLowBitMultiDecoderlayer(
        parameters=layer_weights,
        input_laynorm_weights=input_layer_norm_weights,
        post_attn_layernorm_weights=post_attn_layernorm_weights,
        layer_indexes=layer_indexs,
        cached_cos=cached_cos,
        cached_sin=cached_sin,
        num_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        max_seq_len=args.max_seq_len,
        transpose_value=args.transpose_value_cache
    )

    model.model.multi_decoder = multi_decoder
    print(model)

    with torch.inference_mode():
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        print("finish to load")
        print('input length:', len(input_ids[0]))
        for i in range(3):
            st = time.time()
            output = model.generate(input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict)
            end = time.time()
            if my_rank == 0:
                print(f"First token cost: {model.first_token_time} s, rest tokens cost average: {model.rest_cost_mean} s")
                print(f'Inference time: {end-st} s')
                output_str = tokenizer.decode(output[0], skip_special_tokens=False)
                print('-'*20, 'Prompt', '-'*20)
                print(args.prompt)
                print('-'*20, 'Output', '-'*20)
                print(output_str)
