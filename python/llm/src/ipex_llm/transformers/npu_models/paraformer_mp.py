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
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory



class LowBitMultiEncoderlayer(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        *shapes,
        num_layers: int,
        rms_norm_eps,
        input_layernorm_weights=None,
        input_layernorm_bias=None,
        post_attn_layernorm_weights=None,
        post_attn_layernorm_bias=None,
        fsmn_weight=None,
        qkv_bias=None,
        out_bias=None,
        w1_bias=None,
        w2_bias=None,
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

        self.batch_size = 1
        self.time = 218
        self.size = 512

        input_x = self.create_input_op((self.batch_size, self.time, self.size))
        mask = self.create_input_op((self.batch_size, self.time))

        x = input_x
        
        if input_layernorm_weights is None:
            input_layernorm_weights = []
            input_layernorm_bias = []
            post_attn_layernorm_weights = []
            post_attn_layernorm_bias = []
            fsmn_weight = []
            qkv_bias = []
            out_bias = []
            w1_bias = []
            w2_bias = []
            for i in range(self.num_layers):
                input_layernorm_weights.append(
                    self.create_input_op(
                        (
                            #1,
                            self.size,
                        )
                    )
                )
                input_layernorm_bias.append(
                    self.create_input_op(
                        (
                            #1,
                            self.size,
                        )
                    )
                )
                post_attn_layernorm_weights.append(
                    self.create_input_op(
                        (
                            #1,
                            self.size,
                        )
                    )
                )
                post_attn_layernorm_bias.append(
                    self.create_input_op(
                        (
                            #1,
                            self.size,
                        )
                    )
                )
                fsmn_weight.append(
                    self.create_input_op((512, 512, 11))
                )
                qkv_bias.append(
                    self.create_input_op((1536,))
                )
                out_bias.append(
                    self.create_input_op((512,))
                )
                w1_bias.append(
                    self.create_input_op((2048,))
                )
                w2_bias.append(
                    self.create_input_op((512,))
                )
        else:
            input_layernorm_weights = [self.constant(w) for w in input_layernorm_weights]
            input_layernorm_bias = [self.constant(w) for w in input_layernorm_bias]
            post_attn_layernorm_weights = [self.constant(w) for w in post_attn_layernorm_weights]
            post_attn_layernorm_bias = [self.constant(w) for w in post_attn_layernorm_bias]
            fsmn_weight = [self.constant(w) for w in fsmn_weight]
            qkv_bias = [self.constant(w) for w in qkv_bias]
            out_bias = [self.constant(w) for w in out_bias]
            w1_bias = [self.constant(w) for w in w1_bias]
            w2_bias = [self.constant(w) for w in w2_bias]
        
        
        for i in range(self.num_layers):
            x, mask = self.build_encoder( 
                x=x,
                mask=mask,
                input_layernorm_weight=input_layernorm_weights[i],
                input_layernorm_bias=input_layernorm_bias[i],
                post_attention_layernorm_weight=post_attn_layernorm_weights[i],
                post_attention_layernorm_bias=post_attn_layernorm_bias[i],
                fsmn_weight=fsmn_weight[i],
                qkv_bias=qkv_bias[i],
                out_bias=out_bias[i],
                w1_bias=w1_bias[i],
                w2_bias=w2_bias[i],
            )
            #curr_caches.append((cache))
        
        # define outputs
        x = self.convert_to_fp16(x)
        mask = self.convert_to_fp16(mask)

        print("start compiling")
        self.compile()
    

    def build_encoder(self,
                      x,
                      mask,
                      #cache=None,
                      input_layernorm_weight=None,
                      input_layernorm_bias=None,
                      post_attention_layernorm_weight=None,
                      post_attention_layernorm_bias=None,
                      fsmn_weight=None,
                      qkv_bias=None,
                      out_bias=None,
                      w1_bias=None,
                      w2_bias=None,
        ):
        # EncoderLayerSANM forward
 
        in_feat = 512
        n_feat = 512
        n_head = 4
        hidden_units = 2048
        idim = 512

        stoch_layer_coeff = 1.0
        residual = x
        x = self.paraformer_layer_norm(x, input_layernorm_weight, input_layernorm_bias)

        x = self.self_attn_sanm(x, mask, in_feat, n_feat, n_head, fsmn_weight, qkv_bias, out_bias)
        x = self.eltwise_add(residual, x)

        residual = x
        
        x = self.paraformer_layer_norm(x, post_attention_layernorm_weight, post_attention_layernorm_bias)
        x = self.convert_to_fp16(x)

        x = self.sanm_feed_forward(x, hidden_units, idim, w1_bias, w2_bias)
        x = self.eltwise_add(residual, x)

        x = self.convert_to_fp16(x)

        return x, mask

class FusedLlamaLowBitDecoderlayer(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
        layer_norm_0,
        layer_norm_0_bias,
        layer_norm_1,
        layer_norm_1_bias,
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

        np_dtype = np.uint8

        self.backend_cls_prefill = partial(
            LowBitMultiEncoderlayer,
            num_layers=1,
            rms_norm_eps=rms_norm_eps,
            max_seq_len=max_seq_len,
            input_layernorm_weights=None,
            input_layernorm_bias=None,
            post_attn_layernorm_weights=None,
            post_attn_layernorm_bias=None,
            fsmn_weight=None,
            qkv_bias=None,
            out_bias=None,
            w1_bias=None,
            w2_bias=None,
            mode="prefill",
            transpose_value=self.transpose_value,
            dtype=np_dtype,
        )
        
        self.layer_norm_0 = layer_norm_0
        self.layer_norm_0_bias = layer_norm_0_bias
        self.layer_norm_1 = layer_norm_1
        self.layer_norm_1_bias = layer_norm_1_bias
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
        conv = torch.nn.Conv1d(
            512, 512, 11, stride=1, padding=0, groups=1, bias=False).half()
        print(conv)
        print("conv shape:", conv.weight.shape)
        fsmn_weight = conv.weight

        backend_cls = self.backend_cls_prefill
        inputs = (x.to(torch.float16),
                  masks.to(torch.float16),
                  self.layer_norm_0,
                  self.layer_norm_0_bias,
                  self.layer_norm_1,
                  self.layer_norm_1_bias,
                  fsmn_weight.to(torch.float16),
                  self.qkv_bias,
                  self.out_bias,
                  self.w1_bias,
                  self.w2_bias,
                  )
        

        outputs = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )

        return outputs

def run_prefill(
    model, max_output_len, max_prompt_len, transpose_value_cache, input_queue, result_queue
):

    layer_start = 0
    layer_end = 48

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

        layer_norm_0 = curr_layer.norm1.weight.to(torch.float16)
        layer_norm_0_bias = curr_layer.norm1.bias.to(torch.float16)
        layer_norm_1 = curr_layer.norm2.weight.to(torch.float16)
        layer_norm_1_bias = curr_layer.norm2.bias.to(torch.float16)
        qkv_bias = attn_layer.linear_q_k_v.bias.to(torch.float16)
        out_bias = attn_layer.linear_out.bias.to(torch.float16)
        w1_bias = feed_layer.w_1.bias.to(torch.float16)
        w2_bias = feed_layer.w_2.bias.to(torch.float16)

        new_decoderlayer = FusedLlamaLowBitDecoderlayer(
            weights,
            layer_norm_0=layer_norm_0,
            layer_norm_0_bias=layer_norm_0_bias,
            layer_norm_1=layer_norm_1,
            layer_norm_1_bias=layer_norm_1_bias,
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
        input_layer_norm_weights.append(layer_norm_0)
        post_attn_layernorm_weights.append(layer_norm_1)

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

                xs_pad = encoder_outs[-2]
                masks = encoder_outs[-1]

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
        return xs_pad, masks

    def shutdown(self):
        self.prefill_input_queue.put("stop")
        self.p.join(3)
        if self.p.exitcode is None:
            self.p.kill()

    def __del__(self):
        self.shutdown()


from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8
from funasr.models.transformer.utils.subsampling import TooShortUttError, check_short_utt

def gen_funasr_fused_encoder_forward(prefill_runner):

    def funasr_fused_encoder_forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc = None,
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
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # Prefill runner
        encoder_outs = prefill_runner.forward(xs_pad, masks[0])
        xs_pad, new_masks = encoder_outs[0], encoder_outs[1]

        xs_pad = xs_pad.to(torch.float32)

        if self.normalize_before:
           xs_pad = self.after_norm(xs_pad)


        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens, None

    return funasr_fused_encoder_forward
