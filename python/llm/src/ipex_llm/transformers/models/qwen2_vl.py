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
from transformers.models.qwen2_vl.modeling_qwen2_vl import _prepare_4d_causal_attention_mask_with_cache_position
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache, StaticCache
from transformers import GenerationMixin


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen2VLAttention)


def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    **kwargs,
):
    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    rope_deltas = kwargs.get("rope_deltas", None)
    if attention_mask is not None and position_ids is None:
        if cache_position is None or (cache_position is not None and cache_position[0] == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
        else:
            batch_size, seq_length = input_ids.shape
            delta = (
                cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
            )
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if cache_position[0] != 0:
        pixel_values = None
        pixel_values_videos = None

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            batch_size, sequence_length = input_ids.shape
            device = input_ids.device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    dominant_num = kwargs.get("dominant_num", None)
    contextual_num = kwargs.get("contextual_num", None)

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "rope_deltas": rope_deltas,
            "dominant_num": dominant_num,
            "contextual_num": contextual_num
        }
    )
    return model_inputs


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
    grid_thw: torch.Tensor,
    dominant_num: Optional[int] = None,
    contextual_num: Optional[int] = None,
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # TODO: contextual num
    use_visionzip = dominant_num is not None and dominant_num > 1
    last_layer_attention = None

    total_blk_num = len(self.blocks)
    for idx in range(total_blk_num):
        output_attentions = idx == (total_blk_num - 1) and use_visionzip
        layer_outputs = self.blocks[idx](hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb,
                            output_attentions=output_attentions)
        hidden_states = layer_outputs[0]        # 1564, 1280

        if output_attentions:
            last_layer_attention = layer_outputs[1]     # 16, 1564, 1564

    if use_visionzip:
        # Dominant Visual Tokens
        # TODO: batch dim
        # v1: simple select
        # attention_mean = last_layer_attention.mean(0)       # 1564, 1564
        # attention_mean = attention_mean.mean(0)             # 1564
        # top_k_indices = attention_mean.topk(dominant_num, dim=0).indices

        # v2: select 4 token pairs
        attention_mean = last_layer_attention.mean(0)       # 1564, 1564
        attention_mean = attention_mean.reshape(attention_mean.shape[0] * self.spatial_merge_size ** 2, -1)
        attention_mean = attention_mean.mean(0)             # 391
        top_k_indices = attention_mean.topk(dominant_num, dim=0).indices
        # TODO: get height & width
        # interval_size = 22
        # ranges = [(start, start + interval_size - 1) for start in range(0, 391, interval_size)]

        # # Count the elements in each range
        # counts = []
        # for start, end in ranges:
        #     count = ((top_k_indices >= start) & (top_k_indices <= end)).sum().item()
        #     counts.append((start, end, count))

        top_k_indices_copy = top_k_indices.clone()

        top_k_indices = top_k_indices * 4
        top_k_indices = torch.cat([top_k_indices, top_k_indices + 1, top_k_indices + 2, top_k_indices + 3])

        # v3: select 4 token pairs, another dim (attention mean all equal)
        # attention_mean = last_layer_attention.mean(0)       # 1564, 1564
        # attention_mean = attention_mean.reshape(-1, attention_mean.shape[0] * self.spatial_merge_size ** 2)
        # attention_mean = attention_mean.mean(1)             # 1564
        # top_k_indices = attention_mean.topk(dominant_num // 4, dim=0).indices * 4
        # top_k_indices = torch.cat([top_k_indices, top_k_indices + 1, top_k_indices + 2, top_k_indices + 3])

        mask = torch.ones_like(
            hidden_states[:, 0],
            dtype=torch.bool,
            device=hidden_states.device).scatter_(0, top_k_indices, False)
        
        dominant_tokens = hidden_states.masked_select(
            ~mask.unsqueeze(-1)
        ).view(dominant_num * self.spatial_merge_size ** 2, hidden_states.shape[1])

        hidden_ststes_save = dominant_tokens.to(hidden_states.dtype)
    else:
        hidden_ststes_save = hidden_states
        top_k_indices_copy = None

    return self.merger(hidden_ststes_save), top_k_indices_copy


def get_rope_index(
    self,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    selected_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                # TODO: selected indices batch
                if selected_indices is not None:
                    mask = torch.ones_like(
                        t_index,
                        dtype=torch.bool,
                        device=t_index.device).scatter_(0, selected_indices.to(t_index.device), False)
                    t_index = t_index.masked_select(~mask)
                    h_index = h_index.masked_select(~mask)
                    w_index = w_index.masked_select(~mask)
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if selected_indices is not None:
                position_ids = position_ids[:, :, :llm_positions.shape[1]]
                position_ids = llm_positions.to(position_ids.device)
            else:
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

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
    dominant_num: Optional[int] = None,
    contextual_num: Optional[int] = None,
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
            image_embeds, selected_indices = self.visual(pixel_values, grid_thw=image_grid_thw,
                                                         dominant_num=dominant_num,
                                                         contextual_num=contextual_num)
            if selected_indices is None:
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                # Remove redundant |image_pad| and get selected position ids.
                # v2: previous position id
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw,
                    attention_mask=attention_mask,
                    selected_indices=selected_indices)
                attention_mask = attention_mask[:, :inputs_embeds.shape[1]]

                new_image_pad_num = image_embeds.shape[0]
                image_pad_indices = torch.where(input_ids == self.config.image_token_id)[1]
                image_pad_start_idx = image_pad_indices[0]
                image_pad_end_idx = image_pad_indices[-1]
                new_image_pad_end_idx = image_pad_start_idx + new_image_pad_num
                input_ids = torch.cat([input_ids[:, : new_image_pad_end_idx],
                                    input_ids[:, image_pad_end_idx + 1:]], dim=1)
                # # inputs_embeds = torch.cat([inputs_embeds[:, :new_image_pad_end_idx, :],
                # #                            inputs_embeds[:, image_pad_end_idx + 1:, :]], dim=1)
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                
                
                # v1: random height & width
                # attention_mask = attention_mask[:, :inputs_embeds.shape[1]]
                # new_grid_thw = torch.tensor([[1, 16, 32]])
                # position_ids, rope_deltas = self.get_rope_index(
                #     input_ids, new_grid_thw,
                #     attention_mask=attention_mask)

            torch.xpu.synchronize()
            t2 = time.perf_counter()
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
