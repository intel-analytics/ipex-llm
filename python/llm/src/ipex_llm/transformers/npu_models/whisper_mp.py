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

from typing import Optional, Sequence, List, Union, Any, Tuple
import numpy as np

from transformers.cache_utils import Cache
from ipex_llm.utils.common import invalidInputError
from typing import Optional, List, Generator
import uuid
from functools import partial
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist

from transformers.utils import logging

logger = logging.get_logger(__name__)
from colorama import Fore, Back, Style
import torch.multiprocessing as mp
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from ipex_llm.transformers.npu_models.common import reshape_lm_head_input
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss


class LowBitWhisperMultiEncoderlayer(LLMBaseNNFactory):
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
        self_attn_layer_norm_weights=None,
        self_attn_layer_norm_biases=None,
        final_layer_norm_weights=None,
        final_layer_norm_biases=None,
        v_proj_biases=None,
        q_proj_biases=None,
        out_proj_biases=None,
        fc1_biases=None,
        fc2_biases=None,
        # mode: str = "prefill",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        encoder_ffn_dim,
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        # self.mode = mode
        # self.rms_norm_eps = rms_norm_eps
        self.encoder_ffn_dim = encoder_ffn_dim
        self.transpose_value = transpose_value
        self.num_layers = num_layers

        # if mode == "decode":
        #     self.kv_seq_len = self.max_seq_len + 1
        # else:
        #     self.kv_seq_len = self.seq_len

        self.num_heads = num_heads
        # self.num_key_value_heads = num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # define input, the order self.parameter matters
        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

        # Self Attention
        # if mode == "decode":
        #     attention_mask = self.create_input_op((self.batch_size, 1, 1, self.max_seq_len + 1))
        # else:
        #     attention_mask = self.create_input_op((self.batch_size, 1, self.seq_len, self.seq_len))

        # position_ids = self.create_input_op((self.batch_size, self.seq_len))
        # past_keys = []
        # past_values = []
        # if mode == "decode":
        #     for i in range(num_layers):
        #         past_key = self.create_cache_op(
        #             (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
        #         )
        #         if transpose_value:
        #             past_value = self.create_cache_op(
        #                 (self.batch_size, self.num_key_value_heads, self.head_dim, self.max_seq_len)
        #             )
        #         else:
        #             past_value = self.create_cache_op(
        #                 (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
        #             )
        #         past_keys.append(past_key)
        #         past_values.append(past_value)
        # else:
        #     past_keys = [None] * num_layers
        #     past_values = [None] * num_layers


        if self_attn_layer_norm_weights is None:
            self_attn_layer_norm_weights, self_attn_layer_norm_biases = [], []
            final_layer_norm_weights, final_layer_norm_biases = [], []
            for i in range(num_layers):
                self_attn_layer_norm_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
                self_attn_layer_norm_biases.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
                final_layer_norm_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
                final_layer_norm_biases.append(
                    self.create_input_op(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
        else:
            self_attn_layer_norm_weights = [self.constant(w) for w in self_attn_layer_norm_weights]
            self_attn_layer_norm_biases = [self.constant(w) for w in self_attn_layer_norm_biases]
            final_layer_norm_weights = [self.constant(w) for w in final_layer_norm_weights]
            final_layer_norm_biases = [self.constant(w) for w in final_layer_norm_biases]

        if v_proj_biases is None:
            v_proj_biases, q_proj_biases, out_proj_biases = [], [], []
            fc1_biases, fc2_biases = [], []
            for i in range(num_layers):
                v_proj_biases.append(self.create_input_op((self.hidden_size,)))
                q_proj_biases.append(self.create_input_op((self.hidden_size,)))
                out_proj_biases.append(self.create_input_op((self.hidden_size,)))
                fc1_biases.append(self.create_input_op((self.encoder_ffn_dim,)))
                fc2_biases.append(self.create_input_op((self.hidden_size,)))
        else:
            v_proj_biases = [self.constant(w) for w in v_proj_biases]
            q_proj_biases = [self.constant(w) for w in q_proj_biases]
            out_proj_biases = [self.constant(w) for w in out_proj_biases]
            fc1_biases = [self.constant(w) for w in fc1_biases]
            fc2_biases = [self.constant(w) for w in fc2_biases]
        
        hidden_states = input

        for i in range(num_layers):
            hidden_states = self.build_decoder(
                hidden_states=hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                self_attn_layer_norm_weight=self_attn_layer_norm_weights[i],
                self_attn_layer_norm_bias=self_attn_layer_norm_biases[i],
                final_layer_norm_weight=final_layer_norm_weights[i],
                final_layer_norm_bias=final_layer_norm_biases[i],
                v_proj_bias=v_proj_biases[i],
                q_proj_bias=q_proj_biases[i],
                out_proj_bias=out_proj_biases[i],
                fc1_bias=fc1_biases[i],
                fc2_bias=fc2_biases[i],
            )

        # define outputs
        hidden_states = self.convert_to_fp16(hidden_states)

        print("start compiling")
        self.compile()

    def layer_norm(self, hidden_states, layernorm_weight, layernorm_bias):
        hidden_states = self.convert_to_fp32(hidden_states)
        mean_res = self.reduce_mean(hidden_states, -1, keep_dims=True,)
        variance = self.reduce_mean(
            self.power(hidden_states - mean_res, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(1e-5)
        hidden_states = self.eltwise_div(hidden_states - mean_res, self.sqrt(self.eltwise_add(variance, eps)))
        layernorm_weight = self.convert_to_fp32(layernorm_weight)
        hidden_states = self.eltwise_mul(layernorm_weight, hidden_states)
        layernorm_bias = self.convert_to_fp32(layernorm_bias)
        hidden_states = self.eltwise_add(layernorm_bias, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)
        return hidden_states

    def self_attn(self,
                  hidden_states,
                  attention_mask,
                  layer_head_mask,
                  v_proj_bias,
                  q_proj_bias,
                  out_proj_bias,
        ):
        scaling = self.head_dim**-0.5
        query_states = self.linear(
            hidden_states,
            self.hidden_size,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        query_states = query_states + q_proj_bias # q_proj bias=True
        query_states = query_states * scaling
        query_states = self.reshape(
            query_states, [1, self.seq_len, self.num_heads, self.head_dim]
        )
        query_states = self.transpose(query_states, [0, 2, 1, 3])

        # TODO: add more caese, now only consider self_attention for encoder
        # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = self.linear(
            hidden_states,
            self.hidden_size,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        ) # k_proj bias=False
        key_states = self.reshape(
            key_states, [1, self.seq_len, self.num_heads, self.head_dim]
        )
        key_states = self.transpose(key_states, [0, 2, 1, 3])

        value_states = self.linear(
            hidden_states,
            self.hidden_size,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        ) # v_proj bias=True
        value_states = value_states + v_proj_bias
        value_states = self.reshape(
            value_states, [1, self.seq_len, self.num_heads, self.head_dim]
        )
        value_states = self.transpose(value_states, [0, 2, 1, 3])

        query_states = self.reshape(
            query_states, [1 * self.num_heads, self.seq_len, self.head_dim]
        )
        key_states = self.reshape(
            key_states, [1 * self.num_heads, self.seq_len, self.head_dim]
        )
        value_states = self.reshape(
            value_states, [1 * self.num_heads, self.seq_len, self.head_dim]
        )

        attn_weights = self.matmul(query_states, key_states, False, True)
        attn_weights = self.softmax(attn_weights, -1)
        attn_output = self.matmul(attn_weights, value_states, False, False)
        attn_output = self.reshape(
            attn_output, [1, self.num_heads, self.seq_len, self.head_dim]
        )
        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(
            attn_output, [1, self.seq_len, self.hidden_size]
        )

        attn_output = self.linear(
            attn_output,
            self.hidden_size,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        ) # out_proj bias=True
        attn_output = attn_output + out_proj_bias

        # attn_weights_reshaped is None, past_key_value is None
        return attn_output

    def build_decoder(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        self_attn_layer_norm_weight,
        self_attn_layer_norm_bias,
        final_layer_norm_weight,
        final_layer_norm_bias,
        v_proj_bias,
        q_proj_bias,
        out_proj_bias,
        fc1_bias,
        fc2_bias,
    ):
        residual = hidden_states
        hidden_states = self.reshape(hidden_states, (self.batch_size * self.seq_len, self.hidden_size))

        hidden_states = self.layer_norm(hidden_states, self_attn_layer_norm_weight, self_attn_layer_norm_bias)
        # hidden_states = self.layer_norm(hidden_states, self_attn_layer_norm_weight) # test llama layer norm
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            v_proj_bias=v_proj_bias,
            q_proj_bias=q_proj_bias,
            out_proj_bias=out_proj_bias,
        )
        hidden_states = self.eltwise_add(residual, hidden_states)

        residual = hidden_states

        hidden_states = self.layer_norm(hidden_states, final_layer_norm_weight, final_layer_norm_bias)
        # hidden_states = self.layer_norm(hidden_states, final_layer_norm_weight) # test llama layer norm

        # hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.linear(
            hidden_states, self.encoder_ffn_dim, self.hidden_size, bias=False, wt_dtype=self.dtype
        )
        hidden_states = hidden_states + fc1_bias # fc1 bias=True
        hidden_states = self.gelu(hidden_states)
        # hidden_states = self.swish(hidden_states) # test llama activation

        hidden_states = self.linear(
            hidden_states, self.hidden_size, self.encoder_ffn_dim, bias=False, wt_dtype=self.dtype
        )
        hidden_states = hidden_states + fc2_bias # fc2 bias=True
        
        hidden_states = self.eltwise_add(residual, hidden_states)

        hidden_states = self.convert_to_fp16(hidden_states)

        # TODO: torch.clamp op in https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/whisper/modeling_whisper.py#L785
        return hidden_states


class FusedWhisperLowBitEncoderlayer(torch.nn.Module):
    def __init__(
        self,
        parameters: List[torch.Tensor],
        self_attn_layer_norm_weight,
        self_attn_layer_norm_bias,
        final_layer_norm_weight,
        final_layer_norm_bias,
        v_proj_bias,
        q_proj_bias,
        out_proj_bias,
        fc1_bias,
        fc2_bias,
        num_heads: int,
        # num_key_value_heads: int,
        layer_idx: int,
        encoder_ffn_dim,
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
            LowBitWhisperMultiEncoderlayer,
            num_heads=num_heads,
            # num_key_value_heads=num_key_value_heads,
            num_layers=1,
            self_attn_layer_norm_weights=None,
            self_attn_layer_norm_biases=None,
            final_layer_norm_weights=None,
            final_layer_norm_biases=None,
            v_proj_biases=None,
            q_proj_biases=None,
            out_proj_biases=None,
            fc1_biases=None,
            fc2_biases=None,
            max_seq_len=max_seq_len,
            encoder_ffn_dim=encoder_ffn_dim,
            transpose_value=self.transpose_value,
            dtype=np_dtype,
        )
        self.self_attn_layer_norm_weight = self_attn_layer_norm_weight
        self.self_attn_layer_norm_bias = self_attn_layer_norm_bias
        self.final_layer_norm_weight = final_layer_norm_weight
        self.final_layer_norm_bias = final_layer_norm_bias
        self.v_proj_bias = v_proj_bias
        self.q_proj_bias = q_proj_bias
        self.out_proj_bias = out_proj_bias
        self.fc1_bias = fc1_bias
        self.fc2_bias = fc2_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]

        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16),
                  self.self_attn_layer_norm_weight, self.self_attn_layer_norm_bias,
                  self.final_layer_norm_weight, self.final_layer_norm_bias,
                  self.v_proj_bias, self.q_proj_bias, self.out_proj_bias,
                  self.fc1_bias, self.fc2_bias)

        hidden_states = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )[-1:]
        outputs = (hidden_states,)
        return outputs


def run_prefill(
    model, max_output_len, max_prompt_len, transpose_value_cache, input_queue, result_queue
):

    layer_start = 0
    layer_end = model.config.encoder_layers
    num_heads = model.model.encoder.layers[layer_start].self_attn.num_heads
    # num_key_value_heads = model.model.encoder.layers[layer_start].self_attn.num_key_value_heads
    head_dim = model.model.encoder.layers[layer_start].self_attn.head_dim
    encoder_ffn_dim = model.config.encoder_ffn_dim
    
    encoderlayers = []
    layer_indexs = range(layer_start, layer_end)
    # print(f'scale is {model.model.encoder.layers[0].self_attn.q_proj.scale}')
    for layer_idx in layer_indexs:
        curr_layer = model.model.encoder.layers[layer_idx]
        attn_layer = curr_layer.self_attn

        weights = [
            (attn_layer.q_proj.weight, attn_layer.q_proj.scale),
            (attn_layer.k_proj.weight, attn_layer.k_proj.scale),
            (attn_layer.v_proj.weight, attn_layer.v_proj.scale),
            (attn_layer.out_proj.weight, attn_layer.out_proj.scale),
            (curr_layer.fc1.weight, curr_layer.fc1.scale),
            (curr_layer.fc2.weight, curr_layer.fc2.scale),
        ]

        self_attn_layer_norm_weight = curr_layer.self_attn_layer_norm.weight.to(torch.float16)
        self_attn_layer_norm_bias = curr_layer.self_attn_layer_norm.bias.to(torch.float16)
        final_layer_norm_weight = curr_layer.final_layer_norm.weight.to(torch.float16)
        final_layer_norm_bias = curr_layer.final_layer_norm.bias.to(torch.float16)
        v_proj_bias = attn_layer.v_proj.bias.to(torch.float16)
        q_proj_bias = attn_layer.q_proj.bias.to(torch.float16)
        out_proj_bias = attn_layer.out_proj.bias.to(torch.float16)
        fc1_bias = curr_layer.fc1.bias.to(torch.float16)
        fc2_bias = curr_layer.fc2.bias.to(torch.float16)

        new_encoderlayer = FusedWhisperLowBitEncoderlayer(
            weights,
            self_attn_layer_norm_weight=self_attn_layer_norm_weight,
            self_attn_layer_norm_bias=self_attn_layer_norm_bias,
            final_layer_norm_weight=final_layer_norm_weight,
            final_layer_norm_bias=final_layer_norm_bias,
            v_proj_bias=v_proj_bias,
            q_proj_bias=q_proj_bias,
            out_proj_bias=out_proj_bias,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            num_heads=num_heads,
            # num_key_value_heads=num_key_value_heads,
            layer_idx=layer_idx,
            encoder_ffn_dim=encoder_ffn_dim,
            max_seq_len=max_output_len,
            transpose_value=transpose_value_cache,
        )

        model.model.encoder.layers[layer_idx] = new_encoderlayer
        encoderlayers.append(new_encoderlayer)

    print("finish creating all decode layers in prefill")
    result_queue.put("loading finish")

    while True:

        result = input_queue.get()
        if result == "stop":
            break

        hidden_states, attention_mask, layer_head_mask, output_attentions = result
        with torch.inference_mode():
            for idx, layer in enumerate(encoderlayers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                # print(f'=====layer {idx}, hidden_states is {hidden_states}')
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
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        seq_len = hidden_states.size(1)
        invalidInputError(
            seq_len <= self.max_prompt_len,
            (
                f"seq_len: {seq_len} should be less than or equal"
                " to max_prompt_len {self.max_prompt_len}"
            ),
        )
        args = (hidden_states, attention_mask, layer_head_mask, output_attentions)
        self.prefill_input_queue.put(args)
        hidden_states = self.prefill_result_queue.get()
        return hidden_states

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
def gen_whisper_encoder_forward(prefill_runner):

    def whisper_fused_encoder_forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        hidden_states = prefill_runner.forward(hidden_states,
                                               None,
                                               None,
                                               output_attentions)

        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.layer_norm(hidden_states)
        print(f'=====test encoder result: {hidden_states.shape}_{hidden_states}')
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
        
    return whisper_fused_encoder_forward


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

    #### directly use quantizedlinear
    dist.barrier()

    past_key_value = None

    control = torch.empty((), dtype=torch.int)
    with torch.inference_mode():
        while True:

            dist.broadcast(control, src=0)
            if control.item() == -2:
                break
            elif control.item() == -1:
                attention_mask, encoder_hidden_states, past_key_value = input_queue.get()
            else:
                # hidden_states = torch.empty((1, control.item(), model.config.d_model), dtype=torch.float16)
                hidden_states = torch.empty((1, control.item(), model.config.d_model), dtype=torch.float32)
                dist.recv(hidden_states, src=rank - 1)
                for idx in range(layer_start, layer_end):
                    decoder_layer = model.model.decoder.layers[idx]
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        layer_head_mask=None,
                        cross_attn_layer_head_mask=None,
                        past_key_value=past_key_value,
                        output_attentions=False,
                        use_cache=True,
                    )
                    hidden_states = layer_outputs[0]
                    past_key_value = layer_outputs[1]
                    print(f'test past kv is None: {past_key_value is None}')
                dist.send(hidden_states, dst=(rank + 1) % world_size)


class DecodeRunner:
    def __init__(self, model, max_seq_len, intra_pp=2, inter_pp=2, transpose_value_cache=True):
        self.model = model
        self.max_seq_len = max_seq_len
        self.transpose_value_cache = transpose_value_cache
        world_size = inter_pp + 1
        intra_stages = intra_pp
        num_layers = self.model.model.config.decoder_layers

        port = "54791"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)

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
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        layer_head_mask,
        cross_attn_layer_head_mask,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        if self.cache_past_key_value != past_key_value:
            control = torch.tensor(-1, dtype=torch.int)
            dist.broadcast(control, src=0)
            for i in range(len(self.decoder_processes)):
                self.input_queues[i].put((attention_mask, encoder_hidden_states, past_key_value))

        control = torch.tensor(hidden_states.shape[1], dtype=torch.int)
        dist.broadcast(control, src=0)
        # hidden_states = hidden_states.to(torch.float16)
        hidden_states = hidden_states.to(torch.float32)
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


def gen_whisper_decoder_forward(decode_runner):

    def whisper_fused_decoder_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        from ipex_llm.transformers.npu_models.kv import DynamicFusedNormalCache
        if use_cache and not isinstance(past_key_values, DynamicFusedNormalCache):
            past_key_values = DynamicFusedNormalCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        layer_outputs = decode_runner.forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            layer_head_mask=head_mask,
            cross_attn_layer_head_mask=cross_attn_head_mask,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]

        next_decoder_cache += (layer_outputs[1],)

        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    return whisper_fused_decoder_forward
