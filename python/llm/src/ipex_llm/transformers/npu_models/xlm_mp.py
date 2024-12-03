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
import ctypes
from typing import Optional, Sequence, List, Union, Any, Tuple
from typing import Optional, List, Generator
import uuid
from functools import partial
from colorama import Fore, Back, Style
import math

import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import logging

from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.npu_models.convert import module_optimization
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from intel_npu_acceleration_library.backend.factory import NNFactory
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib

logger = logging.get_logger(__name__)


class LowBitMultiEncoderlayer(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        *shapes,
        num_layers: int,
        rms_norm_eps,
        attn_output_norm_weights=None,
        attn_output_norm_biases=None,
        encoder_output_norm_weights=None,
        encoder_output_norm_biases=None,
        attn_self_query_biases=None,
        attn_self_key_biases=None,
        attn_self_value_biases=None,
        attn_output_dense_biases=None,
        encoder_inter_dense_biases=None,
        encoder_output_dense_biases=None,
        intermediate_size=None,
        num_attention_heads=None,
        hidden_size=None,
        mode: str = "prefill",
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

        self.mode = mode
        self.dtype = dtype
        self.num_layers = num_layers
        self.rms_norm_eps = rms_norm_eps
        self.inter_size = intermediate_size
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))
        attention_mask = self.create_input_op((self.batch_size, 1, 1, self.seq_len))

        hidden_states = input

        if attn_output_norm_weights is None:
            attn_output_norm_weights = []
            attn_output_norm_biases = []
            encoder_output_norm_weights = []
            encoder_output_norm_biases = []
            attn_self_query_biases = []
            attn_self_key_biases = []
            attn_self_value_biases = []
            attn_output_dense_biases = []
            encoder_inter_dense_biases = []
            encoder_output_dense_biases = []
            for i in range(self.num_layers):
                attn_output_norm_weights.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                attn_output_norm_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                encoder_output_norm_weights.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                encoder_output_norm_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                attn_self_query_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                attn_self_key_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                attn_self_value_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                attn_output_dense_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
                encoder_inter_dense_biases.append(
                    self.create_input_op((1, self.inter_size,))
                )
                encoder_output_dense_biases.append(
                    self.create_input_op((1, self.hidden_size,))
                )
        else:
            attn_output_norm_weights = [self.constant(w) for w in attn_output_norm_weights]
            attn_output_norm_biases = [self.constant(w) for w in attn_output_norm_biases]
            encoder_output_norm_weights = [self.constant(w) for w in encoder_output_norm_weights]
            encoder_output_norm_biases = [self.constant(w) for w in encoder_output_norm_biases]
            attn_self_query_biases = [self.constant(w) for w in attn_self_query_biases]
            attn_self_key_biases = [self.constant(w) for w in attn_self_key_biases]
            attn_self_value_biases = [self.constant(w) for w in attn_self_value_biases]
            attn_output_dense_biases = [self.constant(w) for w in attn_output_dense_biases]
            encoder_inter_dense_biases = [self.constant(w) for w in encoder_inter_dense_biases]
            encoder_output_dense_biases = [self.constant(w) for w in encoder_output_dense_biases]

        for i in range(self.num_layers):
            outputs = self.build_encoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_output_norm_weight=attn_output_norm_weights[i],
                attn_output_norm_bias=attn_output_norm_biases[i],
                encoder_output_norm_weight=encoder_output_norm_weights[i],
                encoder_output_norm_bias=encoder_output_norm_biases[i],
                attn_self_query_bias=attn_self_query_biases[i],
                attn_self_key_bias=attn_self_key_biases[i],
                attn_self_value_bias=attn_self_value_biases[i],
                attn_output_dense_bias=attn_output_dense_biases[i],
                encoder_inter_dense_bias=encoder_inter_dense_biases[i],
                encoder_output_dense_bias=encoder_output_dense_biases[i],
            )

        # define outputs
        outputs = self.convert_to_fp32(outputs)

        print("start compiling")
        self.compile()

    def build_encoder(self,
                      hidden_states,
                      attention_mask,
                      attn_output_norm_weight,
                      attn_output_norm_bias,
                      encoder_output_norm_weight,
                      encoder_output_norm_bias,
                      attn_self_query_bias,
                      attn_self_key_bias,
                      attn_self_value_bias,
                      attn_output_dense_bias,
                      encoder_inter_dense_bias,
                      encoder_output_dense_bias,):

        # XLMRobertaAttention
        self_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_bias=attn_self_query_bias,
            key_bias=attn_self_key_bias,
            value_bias=attn_self_value_bias,)

        attention_output = self.self_output(input_tensor=self_outputs,
                                            hidden_states=hidden_states,
                                            output_bias=attn_output_dense_bias,
                                            layer_norm_weight=attn_output_norm_weight,
                                            layer_norm_bias=attn_output_norm_bias,)
        # # XLMRobertaAttention End

        intermediate_output = self.self_intermediate(inter_tensor=attention_output,
                                                     inter_bias=encoder_inter_dense_bias,)

        layer_output = self.encoder_output(input_tensor=intermediate_output,
                                           hidden_states=attention_output,
                                           output_bias=encoder_output_dense_bias,
                                           layer_norm_weight=encoder_output_norm_weight,
                                           layer_norm_bias=encoder_output_norm_bias,)
        outputs = layer_output
        return outputs

    def self_attention(self, hidden_states, attention_mask, query_bias, key_bias, value_bias):
        mixed_query_states = self.linear(
            hidden_states, self.all_head_size, self.hidden_size, bias=False, wt_dtype=self.dtype,
        )
        mixed_query_states = mixed_query_states + query_bias

        key_states = self.linear(
            hidden_states, self.all_head_size, self.hidden_size, bias=False, wt_dtype=self.dtype,
        )
        key_states = key_states + key_bias

        value_states = self.linear(
            hidden_states, self.all_head_size, self.hidden_size, bias=False, wt_dtype=self.dtype,
        )
        value_states = value_states + value_bias

        mixed_query_states = self.reshape(mixed_query_states,
                                          [self.batch_size,
                                           self.seq_len,
                                           self.num_attention_heads,
                                           self.attention_head_size])
        key_states = self.reshape(key_states,
                                  [self.batch_size,
                                   self.seq_len,
                                   self.num_attention_heads,
                                   self.attention_head_size])
        value_states = self.reshape(value_states,
                                    [self.batch_size,
                                     self.seq_len,
                                     self.num_attention_heads,
                                     self.attention_head_size])

        query_states = self.transpose(mixed_query_states, [0, 2, 1, 3])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        value_states = self.transpose(value_states, [0, 2, 1, 3])

        attention_scores = self.matmul(query_states, key_states,
                                       False, True) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = self.eltwise_add(attention_scores, attention_mask)
        attention_probs = self.softmax(attention_scores, -1)

        context_states = self.matmul(attention_probs, value_states, False, False)
        context_states = self.transpose(context_states, [0, 2, 1, 3])
        context_states = self.reshape(context_states,
                                      [self.batch_size, self.seq_len, self.all_head_size])
        context_states = self.convert_to_fp16(context_states)

        return context_states

    def self_output(self,
                    input_tensor,
                    hidden_states,
                    output_bias,
                    layer_norm_weight,
                    layer_norm_bias):
        output_states = self.linear(input_tensor,
                                    self.hidden_size,
                                    self.hidden_size,
                                    bias=False,
                                    wt_dtype=self.dtype,)
        output_states = output_states + output_bias
        output_states = self.eltwise_add(output_states, hidden_states)
        output_states = self.paraformer_layer_norm(output_states,
                                                   layer_norm_weight,
                                                   layer_norm_bias)
        return output_states

    def self_intermediate(self, inter_tensor, inter_bias):
        inter_states = self.linear(inter_tensor,
                                   self.inter_size,
                                   self.hidden_size,
                                   bias=False,
                                   wt_dtype=self.dtype,)
        inter_states = self.convert_to_fp32(inter_states)
        inter_bias = self.convert_to_fp32(inter_bias)
        inter_states = inter_states + inter_bias
        return inter_states

    def encoder_output(self,
                       input_tensor,
                       hidden_states,
                       output_bias,
                       layer_norm_weight,
                       layer_norm_bias):
        input_tensor = self.convert_to_fp16(input_tensor)
        output_states = self.linear(self.gelu(input_tensor),
                                    self.hidden_size,
                                    self.inter_size,
                                    bias=False,
                                    wt_dtype=self.dtype)
        output_states = output_states + output_bias
        output_states = self.eltwise_add(output_states, hidden_states)
        output_states = self.paraformer_layer_norm(output_states,
                                                   layer_norm_weight,
                                                   layer_norm_bias)
        return output_states


class FusedLlamaLowBitDecoderlayer(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
        attn_output_norm_weight,
        attn_output_norm_bias,
        encoder_output_norm_weight,
        encoder_output_norm_bias,
        attn_self_query_bias,
        attn_self_key_bias,
        attn_self_value_bias,
        attn_output_dense_bias,
        encoder_inter_dense_bias,
        encoder_output_dense_bias,
        intermediate_size,
        num_attention_heads,
        hidden_size,
        rms_norm_eps,
        layer_idx,
        max_seq_len=128,
        transpose_value=False,
    ):
        super().__init__()

        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value

        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        self.backend_cls_prefill = partial(
            LowBitMultiEncoderlayer,
            num_layers=1,
            rms_norm_eps=rms_norm_eps,
            attn_output_norm_weights=None,
            attn_output_norm_biases=None,
            encoder_output_norm_weights=None,
            encoder_output_norm_biases=None,
            attn_self_query_biases=None,
            attn_self_key_biases=None,
            attn_self_value_biases=None,
            attn_output_dense_biases=None,
            encoder_inter_dense_biases=None,
            encoder_output_dense_biases=None,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            mode="prefill",
            transpose_value=self.transpose_value,
            dtype=np_dtype,
        )

        self.attn_output_norm_weight = attn_output_norm_weight
        self.attn_output_norm_bias = attn_output_norm_bias
        self.encoder_output_norm_weight = encoder_output_norm_weight
        self.encoder_output_norm_bias = encoder_output_norm_bias
        self.attn_self_query_bias = attn_self_query_bias
        self.attn_self_key_bias = attn_self_key_bias
        self.attn_self_value_bias = attn_self_value_bias
        self.attn_output_dense_bias = attn_output_dense_bias
        self.encoder_inter_dense_bias = encoder_inter_dense_bias
        self.encoder_output_dense_bias = encoder_output_dense_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16),
                  attention_mask.to(torch.float16),
                  self.attn_output_norm_weight.to(torch.float16),
                  self.attn_output_norm_bias.to(torch.float16),
                  self.encoder_output_norm_weight.to(torch.float16),
                  self.encoder_output_norm_bias.to(torch.float16),
                  self.attn_self_query_bias.to(torch.float16),
                  self.attn_self_key_bias.to(torch.float16),
                  self.attn_self_value_bias.to(torch.float16),
                  self.attn_output_dense_bias.to(torch.float16),
                  self.encoder_inter_dense_bias.to(torch.float16),
                  self.encoder_output_dense_bias.to(torch.float16),
                  )

        outputs = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )

        return outputs


def run_prefill(
    model, max_output_len, max_prompt_len, transpose_value_cache, input_queue, result_queue
):

    layer_start = 0
    layer_end = model.config.num_hidden_layers

    decoderlayers = []
    layer_weights = []
    layer_indexs = range(layer_start, layer_end)
    rms_norm_eps = 1e-05

    for layer_idx in layer_indexs:
        curr_layer = model.encoder.layer[layer_idx]
        attn_layer = curr_layer.attention

        weights = [
            (attn_layer.self.query.weight),
            (attn_layer.self.key.weight),
            (attn_layer.self.value.weight),
            (attn_layer.output.dense.weight),
            (curr_layer.intermediate.dense.weight),
            (curr_layer.output.dense.weight),
        ]

        attn_output_norm_weight = attn_layer.output.LayerNorm.weight.to(torch.float16)
        attn_output_norm_bias = attn_layer.output.LayerNorm.bias.to(torch.float16)
        encoder_output_norm_weight = curr_layer.output.LayerNorm.weight.to(torch.float16)
        encoder_output_norm_bias = curr_layer.output.LayerNorm.bias.to(torch.float16)
        attn_self_query_bias = attn_layer.self.query.bias.to(torch.float16)
        attn_self_key_bias = attn_layer.self.key.bias.to(torch.float16)
        attn_self_value_bias = attn_layer.self.value.bias.to(torch.float16)
        attn_output_dense_bias = attn_layer.output.dense.bias.to(torch.float16)
        encoder_inter_dense_bias = curr_layer.intermediate.dense.bias.to(torch.float16)
        encoder_output_dense_bias = curr_layer.output.dense.bias.to(torch.float16)

        new_decoderlayer = FusedLlamaLowBitDecoderlayer(
            weights,
            attn_output_norm_weight=attn_output_norm_weight,
            attn_output_norm_bias=attn_output_norm_bias,
            encoder_output_norm_weight=encoder_output_norm_weight,
            encoder_output_norm_bias=encoder_output_norm_bias,
            attn_self_query_bias=attn_self_query_bias,
            attn_self_key_bias=attn_self_key_bias,
            attn_self_value_bias=attn_self_value_bias,
            attn_output_dense_bias=attn_output_dense_bias,
            encoder_inter_dense_bias=encoder_inter_dense_bias,
            encoder_output_dense_bias=encoder_output_dense_bias,
            intermediate_size=model.config.intermediate_size,
            num_attention_heads=model.config.num_attention_heads,
            hidden_size=model.config.hidden_size,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx,
            max_seq_len=max_output_len,
            transpose_value=transpose_value_cache,
        )

        layer_weights.extend(weights)

        model.encoder.layer[layer_idx] = new_decoderlayer
        decoderlayers.append(new_decoderlayer)

    print("finish creating all decode layers in prefill")
    result_queue.put("loading finish")

    while True:

        result = input_queue.get()
        if result == "stop":
            break

        hidden_states, attention_mask = result
        with torch.inference_mode():
            for encoder_layer in decoderlayers:
                layer_outputs = encoder_layer(
                    hidden_states.to(torch.float16),
                    attention_mask.to(torch.float16),
                )

                hidden_states = layer_outputs
            result_queue.put((hidden_states))


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
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        args = (hidden_states, attention_mask)
        self.prefill_input_queue.put(args)
        outputs = self.prefill_result_queue.get()
        return outputs

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


def gen_xlm_fused_encoder_forward(prefill_runner):

    def xlm_fused_encoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        layer_outputs = prefill_runner.forward(hidden_states, attention_mask)

        layer_outputs = layer_outputs.to(torch.float32)
        hidden_states = layer_outputs

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    return xlm_fused_encoder_forward


class XLMPoolLinear(NNFactory):
    def __init__(
        self,
        input_shape,
        output_channels,
        input_channels,
        device: str = "NPU",
    ):
        super().__init__(False, device)

        # define input
        input_node = self.parameter(input_shape, dtype=np.float16)
        res = self.linear(input_node, output_channels, input_channels, bias=True)

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


class XLMPoolLayer(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        output_channel,
        input_channel,
    ):
        super().__init__()

        self.op_id = str(uuid.uuid4())
        self.parameters = [weight, bias]
        self.backend_cls_pooler = partial(
            XLMPoolLinear,
            output_channels=output_channel,
            input_channels=input_channel,
        )

    def forward(self, hidden_states):
        backend_cls = self.backend_cls_pooler
        hidden_states = hidden_states.to(torch.float16)
        return run_model(hidden_states, self.parameters, backend_cls, self.op_id)


class LayerNorm(NNFactory):
    def __init__(
        self,
        input_shape,
        weight_shape,
        bias_shape,
        eps,
        device: str = "NPU",
    ):
        super().__init__(False, device)

        # define input
        input = self.parameter(input_shape, dtype=np.float16)
        weight = self.parameter(weight_shape, dtype=np.float16)
        bias = self.parameter(bias_shape, dtype=np.float16)

        input = self.convert_to_fp32(input)
        mean_res = self.reduce_mean(input, -1, keep_dims=True,)
        variance = self.reduce_mean(
            self.power(input - mean_res, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(eps)
        input = self.eltwise_div(input - mean_res, self.sqrt(self.eltwise_add(variance, eps)))
        weight = self.convert_to_fp32(weight)
        input = self.eltwise_mul(weight, input)
        bias = self.convert_to_fp32(bias)
        input = self.eltwise_add(bias, input)

        # define outputs
        input = self.convert_to_fp16(input)

        print("start compiling")
        self.compile()


class XLMLayerNorm(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        eps=1e-05,
    ):
        super().__init__()
        self.op_id = str(uuid.uuid4())
        self.parameters = [weight, bias]
        self.backend_cls = partial(
            LayerNorm,
            weight_shape=weight.shape,
            bias_shape=bias.shape,
            eps=eps,
        )

    def forward(self, x):
        x = x.to(torch.float16)
        return run_model(x, self.parameters, self.backend_cls, self.op_id)


@module_optimization
def replace_with_Layernorm(layer, qtype=None, device='NPU',
                           modules_to_not_convert=[], group_size=0, **kwargs):
    if isinstance(layer, torch.nn.LayerNorm):
        return XLMLayerNorm(
            weight=layer.weight.to(torch.float16),
            bias=layer.bias.to(torch.float16),
        )


@module_optimization
def replace_with_FP16Linear(layer, qtype, device, modules_to_not_convert,
                            group_size, imatrix=None):
    from ipex_llm.transformers.npu_models.linear import Linear
    if isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        return Linear(layer.weight, layer.bias)
