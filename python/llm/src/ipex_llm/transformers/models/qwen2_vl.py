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
# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ipex_llm.transformers.models.common import merge_qkv_base, attention_softmax
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_sdp_non_causal
from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicNormalCache
from ipex_llm.utils.common import invalidInputError

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache
from transformers import GenerationMixin


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen2VLAttention)


def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any] = None,
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
        self,
        outputs=outputs,
        model_kwargs=model_kwargs,
        is_encoder_decoder=is_encoder_decoder,
        num_new_tokens=num_new_tokens,
    )

    if model_kwargs.get("use_cache", True):
        cache_num = outputs.past_key_values.seen_tokens
        model_kwargs['cache_position'] = torch.tensor([cache_num])

    if getattr(outputs, "rope_deltas", None) is not None:
        model_kwargs["rope_deltas"] = outputs.rope_deltas

    return model_kwargs


def qwen2_vl_model_forward(
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
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # IPEX-LLM OPT start: kv cache and quantize kv cache
    inputs = input_ids if input_ids is not None else inputs_embeds
    use_cache = True if inputs.device.type == "xpu" else use_cache
    num_heads, num_kv_heads = self.config.num_attention_heads, self.config.num_key_value_heads
    use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.down_proj, inputs,
                                            num_heads, num_kv_heads)
    if use_cache:
        if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        elif not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # IPEX-LLM OPT end

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You cannot specify both input_ids and inputs_embeds at the same time, "
                      "and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                                      device=inputs_embeds.device)

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # IPEX-LLM OPT start: use fused 2D rope
    if (torch.equal(position_ids[0], position_ids[1])
            and torch.equal(position_ids[0], position_ids[2])
            and should_use_fuse_rope(hidden_states, position_ids, False)):
        position_ids = position_ids[0].contiguous()
        position_embeddings = self.rotary_emb.inv_freq
    # IEPX_LLM OPT end

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_vision_get_dtype(self) -> torch.dtype:
    return self.patch_embed.proj.weight.dtype


def qwen2_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None,
    output_attentions: bool = False,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1
                                              ).permute(1, 0, 2, 3).unbind(0)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    # q, k, v: [seq_length, num_heads, head_dim]
    
    # TODO: before rope?
    # raw_key_states = k.clone()

    seq_lens = cu_seqlens.tolist()
    invalidInputError(seq_lens[0] == 0 and seq_lens[-1] == seq_length,
                      "unexpected input")

    if use_sdp_non_causal(self.head_dim, q.device, q.dtype) and not output_attentions:
        # TODO: return attn_weights & attn_output
        image_num = len(seq_lens) - 1
        image_size = seq_lens[1] - seq_lens[0]
        guessed_seq_lens = torch.arange(0, (image_num + 1) * image_size, image_size,
                                        dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        if (guessed_seq_lens == cu_seqlens).all():
            q = q.view(image_num, image_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(image_num, image_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(image_num, image_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # q, k, v: [image_num, num_heads, image_size, head_dim]

            attn_output = scaled_dot_product_attention(
                q, k.contiguous(), v.contiguous(),
                None, False
            )
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(seq_length, self.num_heads, self.head_dim)
            # attn_output: [seq_length, num_heads, head_dim]
        else:
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0).contiguous()
            v = v.transpose(0, 1).unsqueeze(0).contiguous()
            # q, k, v: [1, num_heads, seq_length, head_dim]

            attn_outputs = []
            for i in range(image_num):
                start_idx = seq_lens[i]
                end_idx = seq_lens[i + 1]
                tmp_q = q[:, :, start_idx:end_idx, :]
                tmp_k = k[:, :, start_idx:end_idx, :]
                tmp_v = v[:, :, start_idx:end_idx, :]
                attn_output = scaled_dot_product_attention(
                    tmp_q, tmp_k, tmp_v,
                    None, False
                )
                attn_output = attn_output.permute(0, 2, 1, 3)
                # attn_output: [1, seq_length, num_heads, head_dim]
                attn_outputs.append(attn_output)
            attn_output = torch.cat(attn_outputs, dim=1).squeeze(0)
            # attn_output: [seq_length, num_heads, head_dim]
    else:
        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(seq_lens)):
            attention_mask[..., seq_lens[i - 1]:seq_lens[i], seq_lens[i - 1]:seq_lens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # q, k, v: [num_heads, seq_length, head_dim]

        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = attention_softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        # attn_output: [seq_length, num_heads, head_dim]

    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights#, raw_key_states.mean(1)


def qwen2_vl_vision_block_forward(
    self,
    hidden_states,
    cu_seqlens,
    rotary_pos_emb,
    output_attentions: Optional[bool] = False,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb,
        output_attentions=output_attentions
    )
    hidden_states = residual + hidden_states

    # TODO: uncomment & test
    # r = self._info["r"].pop(0)
    # if r > 0:
    #     self.metric = metric

    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)
    
    return outputs


def qwen2_vit_pretrained_model_forward(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    use_visionzip = True
    last_layer_attention = None

    total_blk_num = len(self.blocks)
    for idx in range(total_blk_num):
        output_attentions = idx == (total_blk_num - 1) and use_visionzip
        layer_outputs = self.blocks[idx](hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb,
                            output_attentions=output_attentions)
        hidden_states = layer_outputs[0]        # 1564, 1280

        if output_attentions:
            last_layer_attention = layer_outputs[1]     # 16, 1564, 1564

    # # TODO: select visionzip hidden states
    # dominant_num = 512
    # contextual_num = 10

    # # Dominant Visual Tokens
    # # TODO: batch dim
    # attention_mean = last_layer_attention.mean(0)       # 1564, 1564
    # attention_mean = attention_mean.mean(0)             # 1564
    # top_k_indices = attention_mean.topk(dominant_num, dim=0).indices

    # mask = torch.ones_like(
    #     hidden_states[:, 0],
    #     dtype=torch.bool,
    #     device=hidden_states.device).scatter_(0, top_k_indices, False)
    
    # dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(dominant_num, hidden_states.shape[1])

    # hidden_ststes_save = dominant_tokens.to(hidden_states.dtype)

    hidden_ststes_save = hidden_states

    return self.merger(hidden_ststes_save)


def qwen2_vl_conditional_generation_forward(
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
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    import time
    
    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            t1 = time.perf_counter()
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw,
                attention_mask=attention_mask)
            
            # # # # TODO: remove redundant image_pad
            # new_image_pad_num = image_embeds.shape[0]
            # image_pad_indices = torch.where(input_ids == self.config.image_token_id)[1]
            # image_pad_start_idx = image_pad_indices[0]
            # image_pad_end_idx = image_pad_indices[-1]
            # new_image_pad_end_idx = image_pad_start_idx + new_image_pad_num
            # input_ids = torch.cat([input_ids[:, : new_image_pad_end_idx],
            #                       input_ids[:, image_pad_end_idx + 1:]], dim=1)
            # # # inputs_embeds = torch.cat([inputs_embeds[:, :new_image_pad_end_idx, :],
            # # #                            inputs_embeds[:, image_pad_end_idx + 1:, :]], dim=1)
            # inputs_embeds = self.model.embed_tokens(input_ids)
            # image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            # image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            # new_grid_thw = torch.tensor([[1, 16, 32]])
            # attention_mask = attention_mask[:, :inputs_embeds.shape[1]]
            # position_ids, rope_deltas = self.get_rope_index(
            #     input_ids, new_grid_thw,
            #     attention_mask=attention_mask)

            torch.xpu.synchronize()
            t2 = time.perf_counter()
            # # image_token_num = image_embeds.shape[0]
            # # dominant = 128
            # # contexual = 10
            # # diff = image_token_num - dominant - contexual
            # # inputs_embeds_token = inputs_embeds.shape[1]
            # # inputs_embeds = inputs_embeds[:, : inputs_embeds_token - diff, :]
            print(inputs_embeds.shape, "time1: ", (t2 - t1) * 1000, " ms")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            # t3 = time.perf_counter()
            # print(inputs_embeds.shape, "pixel_values_videos time2: ", (t3 - t2) * 1000, " ms")

        # if inputs_embeds is None:
        #     inputs_embeds = self.model.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)


    # position_ids = position_ids[:, :, : inputs_embeds.shape[1]]
    # attention_mask = attention_mask[:, : inputs_embeds.shape[1]]
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    # if labels is not None:
    #     # Shift so that tokens < n predict n
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     # Flatten the tokens
    #     loss_fct = CrossEntropyLoss()
    #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #     shift_labels = shift_labels.view(-1)
    #     # Enable model parallelism
    #     shift_labels = shift_labels.to(shift_logits.device)
    #     loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )


def qwen2_vl_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    if position_ids.dim() == 2:
        import xe_addons
        inv_freq = position_embeddings
        xe_addons.rotary_half_inplaced(inv_freq, position_ids, query_states, key_states)
    else:
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == key_states.size(2)
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
