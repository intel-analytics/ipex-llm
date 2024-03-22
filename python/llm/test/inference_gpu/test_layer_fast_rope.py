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
#
# This file is adapted from 
# https://github.com/Dao-AILab/flash-attention/blob/main/tests/layers/test_rotary.py
#
# Copyright (c) 2023, Tri Dao.
#

import os
import pytest
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
from einops import rearrange
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as apply_rotary_pos_emb_llama,
)
from ipex_llm.transformers.layers.rope_embedding import apply_fast_rope_embedding

device = os.environ['DEVICE']
print(f'Running on {device}')
if 'xpu' not in device:
    print(f"The layer.fast_rope test should running on xpu")

# llama-style rotary embedding
@pytest.mark.parametrize("seqlen_offset", [0, 711])
@pytest.mark.parametrize("rotary_emb_fraction", [0.5, 1.0])
def test_rotary(rotary_emb_fraction, seqlen_offset):
    device = "xpu"
    dtype = torch.float16
    rtol, atol = (1e-3, 5e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen_total = 2048
    seqlen = seqlen_total - seqlen_offset
    seqlen_offset = torch.tensor([[seqlen_offset]], device=device)
    nheads = 32
    headdim = 128
    rotary_dim = int(headdim * rotary_emb_fraction)
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
        requires_grad=True
    )
    rotary_llama = LlamaRotaryEmbedding(rotary_dim, seqlen_total, device=device)
    # Doesn't matter what tensor we pass in, rotary_llama only uses the device
    # of the tensor
    cos_llama, sin_llama = rotary_llama(qkv, seq_len=seqlen_total)
    cos_llama, sin_llama = cos_llama.to(dtype=dtype), sin_llama.to(dtype=dtype)
    q_pt = (
        rearrange(qkv[:, :, 0, :, :rotary_dim], "b s h d -> b h s d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k_pt = (
        rearrange(qkv[:, :, 1, :, :rotary_dim], "b s h d -> b h s d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    q_pt_fast = (
        rearrange(qkv[:, :, 0, :, :rotary_dim], "b s h d -> b h s d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k_pt_fast = (
        rearrange(qkv[:, :, 1, :, :rotary_dim], "b s h d -> b h s d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    q_llama, k_llama = apply_rotary_pos_emb_llama(q_pt, k_pt, cos_llama,
                                                  sin_llama, position_ids=seqlen_offset)
    q_fast, k_fast = apply_fast_rope_embedding(q_pt_fast, k_pt_fast,
                                               position_ids=seqlen_offset,
                                               model_family="llama")
    assert torch.allclose(
        rearrange(q_llama, "b h s d -> b s h d"),
        rearrange(q_fast, "b h s d -> b s h d"), rtol=rtol, atol=atol
    )
    assert torch.allclose(
        rearrange(k_llama, "b h s d -> b s h d"),
        rearrange(k_fast, "b h s d -> b s h d"), rtol=rtol, atol=atol
    )

    g = torch.randn_like(q_fast)
    q_fast.backward(g)
    k_fast.backward(g)
    q_llama.backward(g)
    k_llama.backward(g)

    assert torch.allclose(
        q_pt.grad,
        q_pt_fast.grad,
        rtol=rtol,
        atol=atol,
    )

    assert torch.allclose(
        k_pt.grad,
        k_pt_fast.grad,
        rtol=rtol,
        atol=atol,
    )

if __name__ == "__main__":
    pytest.main([__file__])
