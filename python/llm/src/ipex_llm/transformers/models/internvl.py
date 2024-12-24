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
# https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# which is licensed under MIT:
#
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#

import torch
from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import use_sdp_non_causal


def _get_pos_embed(self, pos_embed, H, W):
    target_dtype = pos_embed.dtype
    device = pos_embed.device
    pos_embed = pos_embed.float().reshape(
        1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1
    ).permute(0, 3, 1, 2)
    # ipex-llm change start: call interpolate on CPU to fix bug
    pos_embed = torch.nn.functional.interpolate(
        pos_embed.to('cpu'), size=(H, W), mode='bicubic', align_corners=False
    ).reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype).to(device)
    # ipex-llm changes end
    return pos_embed


def internvl_chat(self, tokenizer, pixel_values, question, generation_config,
                  history=None, return_history=False, num_patches_list=None,
                  IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                  IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    invalidInputError(pixel_values is None or len(pixel_values) == sum(num_patches_list),
                      "wrong num_patches_list length")

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    template = self.get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = (IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                        + IMG_END_TOKEN)
        query = query.replace('<image>', image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors='pt')

    # ipex-llm changes start: move input_ids and attention_mask to xpu
    input_ids = model_inputs['input_ids'].to(self.device)
    attention_mask = model_inputs['attention_mask'].to(self.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=self.dtype, device=self.device)
    # ipex-llm changes end

    generation_config['eos_token_id'] = eos_token_id
    generation_output = self.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split(template.sep)[0].strip()
    history.append((question, response))
    if return_history:
        return response, history
    else:
        query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
        query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        if verbose:
            print(query_to_print, response)
        return response


def internvl_batch_chat(self, tokenizer, pixel_values, questions, generation_config,
                        num_patches_list=None, history=None, return_history=False,
                        IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
    invalidInputError(history is None and not return_history,
                      'Now multi-turn chat is not supported in batch_chat.')

    if image_counts is not None:
        num_patches_list = image_counts
        print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    queries = []
    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        template = self.get_conv_template(self.template)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                        + IMG_END_TOKEN)
        query = query.replace('<image>', image_tokens, 1)
        queries.append(query)

    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(queries, return_tensors='pt', padding=True)

    # ipex-llm changes start: move input_ids and attention_mask to xpu
    input_ids = model_inputs['input_ids'].to(self.device)
    attention_mask = model_inputs['attention_mask'].to(self.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=self.dtype, device=self.device)
    # ipex-llm changes end

    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    generation_config['eos_token_id'] = eos_token_id
    generation_output = self.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    responses = [response.split(template.sep)[0].strip() for response in responses]
    return responses


def intern_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    if use_sdp_non_causal(self.head_dim, q.device, q.dtype):
        x = scaled_dot_product_attention(
            q, k.contiguous(), v.contiguous(),
            None, False, self.scale
        )
    else:
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
