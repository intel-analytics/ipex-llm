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

import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
from transformers.cache_utils import Cache
from ipex_llm.utils.common import invalidInputError
from transformers.utils import logging

import torch.multiprocessing as mp
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib

from funasr.models.scama import utils as myutils
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling, Conv2dSubsampling2, \
    Conv2dSubsampling6, Conv2dSubsampling8, TooShortUttError, check_short_utt

logger = logging.get_logger(__name__)


class LowBitMultiEncoderlayer(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        *shapes,
        num_layers: int,
        rms_norm_eps,
        layer_norm_0_weights=None,
        layer_norm_0_biases=None,
        layer_norm_1_weights=None,
        layer_norm_1_biases=None,
        fsmn_weights=None,
        qkv_biases=None,
        out_biases=None,
        w1_biases=None,
        w2_biases=None,
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
        self.batch_size, self.time, self.size = hidden_shape

        input_x = self.create_input_op((self.batch_size, self.time, self.size))
        mask = self.create_input_op((self.batch_size, self.time))

        x = input_x

        if layer_norm_0_weights is None:
            layer_norm_0_weights = []
            layer_norm_0_biases = []
            layer_norm_1_weights = []
            layer_norm_1_biases = []
            fsmn_weights = []
            qkv_biases = []
            out_biases = []
            w1_biases = []
            w2_biases = []
            for i in range(self.num_layers):
                layer_norm_0_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.size,
                        )
                    )
                )
                layer_norm_0_biases.append(
                    self.create_input_op(
                        (
                            1,
                            self.size,
                        )
                    )
                )
                layer_norm_1_weights.append(
                    self.create_input_op(
                        (
                            1,
                            self.size,
                        )
                    )
                )
                layer_norm_1_biases.append(
                    self.create_input_op(
                        (
                            1,
                            self.size,
                        )
                    )
                )
                fsmn_weights.append(
                    self.create_input_op((512, 1, 1, 11))
                )
                qkv_biases.append(
                    self.create_input_op((1536,))
                )
                out_biases.append(
                    self.create_input_op((512,))
                )
                w1_biases.append(
                    self.create_input_op((2048,))
                )
                w2_biases.append(
                    self.create_input_op((512,))
                )
        else:
            layer_norm_0_weights = [self.constant(w) for w in layer_norm_0_weights]
            layer_norm_0_biases = [self.constant(w) for w in layer_norm_0_biases]
            layer_norm_1_weights = [self.constant(w) for w in layer_norm_1_weights]
            layer_norm_1_biases = [self.constant(w) for w in layer_norm_1_biases]
            fsmn_weights = [self.constant(w) for w in fsmn_weights]
            qkv_biases = [self.constant(w) for w in qkv_biases]
            out_biases = [self.constant(w) for w in out_biases]
            w1_biases = [self.constant(w) for w in w1_biases]
            w2_biases = [self.constant(w) for w in w2_biases]

        for i in range(self.num_layers):
            x, mask = self.build_encoder(
                x=x,
                mask=mask,
                layer_norm_0_weight=layer_norm_0_weights[i],
                layer_norm_0_bias=layer_norm_0_biases[i],
                layer_norm_1_weight=layer_norm_1_weights[i],
                layer_norm_1_bias=layer_norm_1_biases[i],
                fsmn_weight=fsmn_weights[i],
                qkv_bias=qkv_biases[i],
                out_bias=out_biases[i],
                w1_bias=w1_biases[i],
                w2_bias=w2_biases[i],
            )

        # define outputs
        x = self.convert_to_fp32(x)
        mask = self.convert_to_fp32(mask)

        print("start compiling")
        self.compile()

    def build_encoder(self,
                      x,
                      mask,
                      layer_norm_0_weight,
                      layer_norm_0_bias,
                      layer_norm_1_weight,
                      layer_norm_1_bias,
                      fsmn_weight,
                      qkv_bias,
                      out_bias,
                      w1_bias,
                      w2_bias,
                      ):

        # EncoderLayerSANM forward
        in_feat = 512
        n_feat = 512
        n_head = 4
        hidden_units = 2048
        idim = 512

        stoch_layer_coeff = 1.0
        residual = x
        x = self.paraformer_layer_norm(x, layer_norm_0_weight, layer_norm_0_bias)
        tmp = x

        x = self.self_attn_sanm(x, mask, in_feat, n_feat, n_head, fsmn_weight, qkv_bias, out_bias)

        x = stoch_layer_coeff * x
        x = self.eltwise_add(residual, x)

        residual = x
        x = self.paraformer_layer_norm(x, layer_norm_1_weight, layer_norm_1_bias)

        x = self.sanm_feed_forward(x, hidden_units, idim, w1_bias, w2_bias)
        x = self.eltwise_add(residual, x)
        x = stoch_layer_coeff * x

        return x, mask


class FusedLlamaLowBitDecoderlayer(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
        layer_norm_0_weight,
        layer_norm_0_bias,
        layer_norm_1_weight,
        layer_norm_1_bias,
        fsmn_weight,
        qkv_bias,
        out_bias,
        w1_bias,
        w2_bias,
        rms_norm_eps,
        layer_idx: int,
        max_seq_len: int = 128,
        transpose_value: bool = False,
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
            layer_norm_0_weights=None,
            layer_norm_0_biases=None,
            layer_norm_1_weights=None,
            layer_norm_1_biases=None,
            fsmn_weights=None,
            qkv_biases=None,
            out_biases=None,
            w1_biases=None,
            w2_biases=None,
            mode="prefill",
            transpose_value=self.transpose_value,
            dtype=np_dtype,
        )

        self.layer_norm_0_weight = layer_norm_0_weight
        self.layer_norm_0_bias = layer_norm_0_bias
        self.layer_norm_1_weight = layer_norm_1_weight
        self.layer_norm_1_bias = layer_norm_1_bias
        self.fsmn_weight = fsmn_weight
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias
        self.w1_bias = w1_bias
        self.w2_bias = w2_bias

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        backend_cls = self.backend_cls_prefill
        inputs = (x.to(torch.float16),
                  masks.to(torch.float16),
                  self.layer_norm_0_weight.to(torch.float16),
                  self.layer_norm_0_bias.to(torch.float16),
                  self.layer_norm_1_weight.to(torch.float16),
                  self.layer_norm_1_bias.to(torch.float16),
                  self.fsmn_weight.to(torch.float16),
                  self.qkv_bias.to(torch.float16),
                  self.out_bias.to(torch.float16),
                  self.w1_bias.to(torch.float16),
                  self.w2_bias.to(torch.float16),
                  )

        outputs = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )

        return outputs


def run_prefill(
    model, max_output_len, max_prompt_len, transpose_value_cache, input_queue, result_queue
):

    layer_start = 0
    layer_end = 30

    deocderlayers = []
    layer_weights = []
    conv_weights = []
    input_layer_norm_weights = []
    post_attn_layernorm_weights = []
    layer_indexs = range(layer_start, layer_end)
    rms_norm_eps = 1e-12

    for layer_idx in layer_indexs:
        curr_layer = model.model.encoder.encoders[layer_idx]
        attn_layer = curr_layer.self_attn
        feed_layer = curr_layer.feed_forward

        weights = [
            (attn_layer.linear_q_k_v.weight, attn_layer.linear_q_k_v.scale),
            (attn_layer.linear_out.weight, attn_layer.linear_out.scale),
            (feed_layer.w_1.weight, feed_layer.w_1.scale),
            (feed_layer.w_2.weight, feed_layer.w_2.scale),
        ]

        layer_norm_0_weight = curr_layer.norm1.weight.to(torch.float16)
        layer_norm_0_bias = curr_layer.norm1.bias.to(torch.float16)
        layer_norm_1_weight = curr_layer.norm2.weight.to(torch.float16)
        layer_norm_1_bias = curr_layer.norm2.bias.to(torch.float16)
        fsmn_weight = attn_layer.fsmn_block.weight.to(torch.float16)
        qkv_bias = attn_layer.linear_q_k_v.bias.to(torch.float16)
        out_bias = attn_layer.linear_out.bias.to(torch.float16)
        w1_bias = feed_layer.w_1.bias.to(torch.float16)
        w2_bias = feed_layer.w_2.bias.to(torch.float16)

        new_decoderlayer = FusedLlamaLowBitDecoderlayer(
            weights,
            layer_norm_0_weight=layer_norm_0_weight,
            layer_norm_0_bias=layer_norm_0_bias,
            layer_norm_1_weight=layer_norm_1_weight,
            layer_norm_1_bias=layer_norm_1_bias,
            fsmn_weight=fsmn_weight,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx,
            max_seq_len=max_output_len,
            transpose_value=transpose_value_cache,
        )

        layer_weights.extend(weights)

        model.model.encoder.encoders[layer_idx] = new_decoderlayer
        deocderlayers.append(new_decoderlayer)

    print("finish creating all decode layers in prefill")
    result_queue.put("loading finish")

    while True:

        result = input_queue.get()
        if result == "stop":
            break

        xs_pad, masks = result
        with torch.inference_mode():
            for encoder_layer in deocderlayers:
                encoder_outs = encoder_layer(
                    xs_pad, masks
                )

                xs_pad = encoder_outs[0]
                masks = encoder_outs[1]

            result_queue.put((xs_pad, masks))


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
        xs_pad,
        masks,
        **kwargs,
    ):
        args = (xs_pad, masks)
        self.prefill_input_queue.put(args)
        xs_pad, masks = self.prefill_result_queue.get()
        xs_pad = xs_pad.to(torch.float32)
        masks = masks.to(torch.float32)
        return xs_pad, masks

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


class LowBitMultiDecoderlayer(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        mask_shape: Sequence[int],
        memory_shape: Sequence[int],
        memory_mask_shape: Sequence[int],
        *shapes,
        layer_norm_0_weights=None,
        layer_norm_0_biases=None,
        layer_norm_1_weights=None,
        layer_norm_1_biases=None,
        layer_norm_2_weights=None,
        layer_norm_2_biases=None,
        q_biases=None,
        kv_biases=None,
        out_biases=None,
        w1_biases=None,
        feed_norm_weights=None,
        feed_norm_biases=None,
        fsmn_weights=None,
        num_layers: int,
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

        self.x_bsz, self.x_time, self.x_size = hidden_shape
        self.x_mask_bsz, self.x_mask_time, self.x_mask_size = mask_shape
        self.mem_bsz, self.mem_time, self.mem_size = memory_shape
        self.mem_mask_bsz, self.mem_mask_time, self.mem_mask_size = memory_mask_shape

        input = self.create_input_op((self.x_bsz, self.x_time, self.x_size))
        tgt_mask = self.create_input_op((self.x_mask_bsz, self.x_mask_time, self.x_mask_size))
        memory = self.create_input_op((self.mem_bsz, self.mem_time, self.mem_size))
        memory_mask = self.create_input_op((self.mem_mask_bsz,
                                            self.mem_mask_time,
                                            self.mem_mask_size))

        layer_norm_0_weights = [self.constant(w) for w in layer_norm_0_weights]
        layer_norm_0_biases = [self.constant(w) for w in layer_norm_0_biases]
        layer_norm_1_weights = [self.constant(w) for w in layer_norm_1_weights]
        layer_norm_1_biases = [self.constant(w) for w in layer_norm_1_biases]
        layer_norm_2_weights = [self.constant(w) for w in layer_norm_2_weights]
        layer_norm_2_biases = [self.constant(w) for w in layer_norm_2_biases]
        q_biases = [self.constant(w) for w in q_biases]
        kv_biases = [self.constant(w) for w in kv_biases]
        out_biases = [self.constant(w) for w in out_biases]
        w1_biases = [self.constant(w) for w in w1_biases]
        feed_norm_weights = [self.constant(w) for w in feed_norm_weights]
        feed_norm_biases = [self.constant(w) for w in feed_norm_biases]
        fsmn_weights = [self.constant(w) for w in fsmn_weights]

        x = input
        for i in range(self.num_layers):
            x, tgt_mask, memory, memory_mask = self.build_decoder(
                x=x,
                tgt_mask=tgt_mask,
                memory=memory,
                memory_mask=memory_mask,
                norm1_weight=layer_norm_0_weights[i],
                norm1_bias=layer_norm_0_biases[i],
                norm2_weight=layer_norm_1_weights[i],
                norm2_bias=layer_norm_1_biases[i],
                norm3_weight=layer_norm_2_weights[i],
                norm3_bias=layer_norm_2_biases[i],
                q_bias=q_biases[i],
                kv_bias=kv_biases[i],
                out_bias=out_biases[i],
                w1_bias=w1_biases[i],
                feed_norm_weight=feed_norm_weights[i],
                feed_norm_bias=feed_norm_biases[i],
                fsmn_weight=fsmn_weights[i],
            )

        # define outputs
        x = self.convert_to_fp16(x)
        tgt_mask = self.convert_to_fp16(tgt_mask)
        memory = self.convert_to_fp16(memory)
        memory_mask = self.convert_to_fp16(memory_mask)

        print("start compiling")
        self.compile()

    def build_decoder(self,
                      x,
                      tgt_mask,
                      memory,
                      memory_mask=None,
                      norm1_weight=None,
                      norm1_bias=None,
                      norm2_weight=None,
                      norm2_bias=None,
                      norm3_weight=None,
                      norm3_bias=None,
                      fsmn_weight=None,
                      q_bias=None,
                      kv_bias=None,
                      out_bias=None,
                      w1_bias=None,
                      feed_norm_weight=None,
                      feed_norm_bias=None,
                      ):

        in_feat = 512
        n_feat = 512
        n_head = 4
        idim = 512
        hidden_units = 2048
        stoch_layer_coeff = 1.0

        residual = x

        # norm1
        x = self.paraformer_layer_norm(x, norm1_weight, norm1_bias)
        x = self.feed_forward_sanm_decoder(x, w1_bias, feed_norm_weight, feed_norm_bias)

        # norm2
        x = self.paraformer_layer_norm(x, norm2_weight, norm2_bias)
        x, _ = self.multihead_attn_sanm_decoder(x, tgt_mask, fsmn_weight)
        x = self.eltwise_add(residual, x)
        residual = x

        # norm3
        x = self.paraformer_layer_norm(x, norm3_weight, norm3_bias)
        x_src_attn = self.sanm_cross_attn(x, memory, memory_mask,
                                          q_bias, kv_bias, out_bias, n_feat, n_head)

        x = self.eltwise_add(residual, x_src_attn)
        x = self.convert_to_fp16(x)

        return x, tgt_mask, memory, memory_mask

    def run_multi_decoders(inputs, decoders, models_ptr=None):
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

        x, tgt_mask, memory, memory_mask = decoders[-1].torch_out
        return x, tgt_mask, memory, memory_mask


class FusedLlamaLowBitMultiDecoderlayer(torch.nn.Module):

    def __init__(
        self,
        parameters,
        layer_norm_0_weights,
        layer_norm_0_biases,
        layer_norm_1_weights,
        layer_norm_1_biases,
        layer_norm_2_weights,
        layer_norm_2_biases,
        q_biases,
        kv_biases,
        out_biases,
        w1_biases,
        feed_norm_weights,
        feed_norm_biases,
        fsmn_weights,
        layer_indexes,
        intra_stages,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        do_print: bool = True,
        x_shape=None,
        x_mask_shape=None,
        memory_shape=None,
        memory_mask_shape=None,
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

        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
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
            decoder = LowBitMultiDecoderlayer(
                x_shape,
                x_mask_shape,
                memory_shape,
                memory_mask_shape,
                layer_norm_0_weights=layer_norm_0_weights[start:end],
                layer_norm_0_biases=layer_norm_0_biases[start:end],
                layer_norm_1_weights=layer_norm_1_weights[start:end],
                layer_norm_1_biases=layer_norm_1_biases[start:end],
                layer_norm_2_weights=layer_norm_2_weights[start:end],
                layer_norm_2_biases=layer_norm_2_biases[start:end],
                q_biases=q_biases[start:end],
                kv_biases=kv_biases[start:end],
                out_biases=out_biases[start:end],
                w1_biases=w1_biases[start:end],
                feed_norm_weights=feed_norm_weights[start:end],
                feed_norm_biases=feed_norm_biases[start:end],
                fsmn_weights=fsmn_weights[start:end],
                num_layers=end - start,
                max_seq_len=max_seq_len,
                mode="decode",
                transpose_value=self.transpose_value,
                dtype=np_dtype,
            )
            self.backend_decoders.append(decoder)

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            self.backend_decoders[i].set_weights(self.op_id, op_parameters[start * 5:end * 5])

    def forward(
        self,
        x,
        tgt_mask,
        memory,
        memory_mask=None,
        cache=None,
        **kwargs,
    ) -> torch.Tensor:

        inputs = (
            x.to(torch.float16),
            tgt_mask,
            memory.to(torch.float16),
            memory_mask,
        )

        x, tgt_mask, memory, memory_mask = LowBitMultiDecoderlayer.run_multi_decoders(
            inputs,
            decoders=self.backend_decoders)

        if self.do_print:
            print("outputs:", x)

        outputs = (x, tgt_mask, memory, memory_mask)
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

    deocderlayers = []
    layer_weights = []
    layer_norm_0_weights = []
    layer_norm_0_biases = []
    layer_norm_1_weights = []
    layer_norm_1_biases = []
    layer_norm_2_weights = []
    layer_norm_2_biases = []
    fsmn_weights = []
    q_biases = []
    kv_biases = []
    out_biases = []
    w1_biases = []
    feed_norm_weights = []
    feed_norm_biases = []

    layer_indexs = range(layer_start, layer_end)

    for layer_idx in layer_indexs:
        curr_layer = model.model.decoder.decoders[layer_idx]
        attn_layer = curr_layer.self_attn
        src_attn_layer = curr_layer.src_attn
        feed_layer = curr_layer.feed_forward

        weights = [
            (feed_layer.w_1.weight, feed_layer.w_1.scale),
            (feed_layer.w_2.weight, feed_layer.w_2.scale),
            (src_attn_layer.linear_q.weight, src_attn_layer.linear_q.scale),
            (src_attn_layer.linear_k_v.weight, src_attn_layer.linear_k_v.scale),
            (src_attn_layer.linear_out.weight, src_attn_layer.linear_out.scale),
        ]

        layer_weights.extend(weights)
        # norm_0
        layer_norm_0_weights.append(curr_layer.norm1.weight.to(torch.float16))
        layer_norm_0_biases.append(curr_layer.norm1.bias.to(torch.float16))
        # norm_1
        layer_norm_1_weights.append(curr_layer.norm2.weight.to(torch.float16))
        layer_norm_1_biases.append(curr_layer.norm2.bias.to(torch.float16))
        # norm_2
        layer_norm_2_weights.append(curr_layer.norm3.weight.to(torch.float16))
        layer_norm_2_biases.append(curr_layer.norm3.bias.to(torch.float16))
        # linear_q
        q_biases.append(src_attn_layer.linear_q.bias.to(torch.float16))
        # linear_kv
        kv_biases.append(src_attn_layer.linear_k_v.bias.to(torch.float16))
        # linear_out
        out_biases.append(src_attn_layer.linear_out.bias.to(torch.float16))
        # linear_w1
        w1_biases.append(feed_layer.w_1.bias.to(torch.float16))
        # feed_norm
        feed_norm_weights.append(feed_layer.norm.weight.to(torch.float16))
        feed_norm_biases.append(feed_layer.norm.bias.to(torch.float16))
        # conv weights
        fsmn_weights.append(attn_layer.fsmn_block.weight.view(512, 1, 1, 11).to(torch.float16))

    dist.barrier()
    control = torch.empty((), dtype=torch.int)

    with torch.inference_mode():
        while True:
            dist.broadcast(control, src=0)
            if control.item() == -2:
                break
            elif control.item() == -1:
                x, tgt_mask, memory, memory_mask = input_queue.get()
            else:
                dist.recv(x, src=rank - 1)
                t1 = time.perf_counter()

                multi_decoder = FusedLlamaLowBitMultiDecoderlayer(
                    parameters=layer_weights,
                    layer_norm_0_weights=layer_norm_0_weights,
                    layer_norm_0_biases=layer_norm_0_biases,
                    layer_norm_1_weights=layer_norm_1_weights,
                    layer_norm_1_biases=layer_norm_1_biases,
                    layer_norm_2_weights=layer_norm_2_weights,
                    layer_norm_2_biases=layer_norm_2_biases,
                    q_biases=q_biases,
                    kv_biases=kv_biases,
                    out_biases=out_biases,
                    w1_biases=w1_biases,
                    feed_norm_weights=feed_norm_weights,
                    feed_norm_biases=feed_norm_biases,
                    fsmn_weights=fsmn_weights,
                    layer_indexes=layer_indexs,
                    intra_stages=intra_stages,
                    max_seq_len=max_seq_len,
                    transpose_value=transpose_value_cache,
                    do_print=True,
                    x_shape=list(x.shape),
                    x_mask_shape=list(tgt_mask.shape),
                    memory_shape=list(memory.shape),
                    memory_mask_shape=list(memory_mask.shape),
                )

                layer_outputs = multi_decoder(
                    x=x,
                    tgt_mask=tgt_mask,
                    memory=memory,
                    memory_mask=memory_mask,
                    cache=None,
                )

                t2 = time.perf_counter()
                x = layer_outputs[0]
                t3 = time.perf_counter()
                dist.send(x, dst=(rank + 1) % world_size)
                t4 = time.perf_counter()
                tgt_mask = layer_outputs[1]
                memory = layer_outputs[2]
                memory_mask = layer_outputs[3]


class DecodeRunner:
    def __init__(self, model, max_seq_len, intra_pp=2, inter_pp=2, transpose_value_cache=True):
        self.model = model
        self.max_seq_len = max_seq_len
        self.transpose_value_cache = transpose_value_cache
        world_size = inter_pp + 1
        intra_stages = intra_pp
        num_layers = 16

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
        self.prev_cache = None

    def forward(
        self,
        x,
        tgt_mask,
        memory,
        memory_mask=None,
        cache=None,
        **kwargs,
    ):
        t0 = time.perf_counter()
        x = x.to(torch.float16)

        if self.prev_cache is None:
            control = torch.tensor(-1, dtype=torch.int)
            dist.broadcast(control, src=0)
            for i in range(len(self.decoder_processes)):
                self.input_queues[i].put((x, tgt_mask, memory, memory_mask))

        dist.broadcast(self.forward_signal, src=0, async_op=True)
        dist.send(x, dst=1)
        dist.recv(x, src=self.world_size - 1)

        t1 = time.perf_counter()
        return x, tgt_mask, memory, memory_mask, cache

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


def gen_funasr_fused_encoder_forward(prefill_runner):

    def funasr_fused_encoder_forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            invalidInputError(
                not short_status,
                (
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    f"(it needs more than {limit_size} frames), return empty results"
                ),
            )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # Prefill runner
        encoder_outs = prefill_runner.forward(xs_pad, masks[0])
        xs_pad, new_masks = encoder_outs[0], encoder_outs[1]

        encoders_suffix = self.encoders[31:49]
        encoder_outs = encoders_suffix(xs_pad, masks[0])
        xs_pad, new_masks, mm = encoder_outs[0], encoder_outs[1], encoder_outs[2]
        xs_pad = xs_pad.to(torch.float32)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens, None

    return funasr_fused_encoder_forward


def gen_funasr_fused_decoder_forward(decode_runner):

    def funasr_fused_decoder_forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        chunk_mask: torch.Tensor = None,
        return_hidden: bool = False,
        return_both: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        tgt = ys_in_pad
        tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]
        if chunk_mask is not None:
            memory_mask = memory_mask * chunk_mask
            if tgt_mask.size(1) != memory_mask.size(1):
                memory_mask = torch.cat((memory_mask, memory_mask[:, -2:-1, :]), dim=1)

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = decode_runner.forward(x,
                                                                    tgt_mask,
                                                                    memory,
                                                                    memory_mask)
        x = x.to(torch.float32)

        x, tgt_mask, memory, memory_mask, _ = self.decoders3(x,
                                                             tgt_mask,
                                                             memory,
                                                             memory_mask)
        if self.normalize_before:
            hidden = self.after_norm(x)

        olens = tgt_mask.sum(1)
        if self.output_layer is not None and return_hidden is False:
            x = self.output_layer(hidden)
            return x, olens
        if return_both:
            x = self.output_layer(hidden)
            return x, hidden, olens
        return hidden, olens

    return funasr_fused_decoder_forward
