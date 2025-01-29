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
# Convert a GPTQ quantized LLaMA model to a ggml compatible file
# Based on: https://github.com/ggerganov/llama.cpp
#           /blob/20a1a4e09c522a80e2a0db51643d25fa38326065/convert-gptq-to-ggml.py
# Current supported GPTQ model: 4bits, no act-order, no safetensors.
#
import os
import re
import sys
import json
import warnings
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from bigdl.llm.utils.common.log4Error import invalidInputError


def write_header(fout, shape, dst_name, ftype_cur):
    sname = dst_name.encode('utf-8')
    fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    # fout.seek((fout.tell() + 31) & -32)   # not in bloom


def convert_non_q4(src_name, dst_name, model, fout):
    v = model[src_name]
    shape = v.shape
    print("Processing non-Q4 variable: " + src_name +
          " with shape: ", shape, " and type: ", v.dtype)
    # if len(shape) == 1:
    #     print("  Converting to float32")
    #     v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(fout, shape, dst_name, ftype_cur)

    # data
    v.numpy().tofile(fout)


def convert_fp_to_q4(src_name, dst_name, model, fout):
    v = model[src_name]
    shape = v.shape
    print("Processing non-Q4 variable: " + src_name +
          " with shape: ", shape, " and type: ", v.dtype)
    print("Converting to int4")
    v = v.to(torch.float32)

    group_size = 64
    ftype = 3  # Q4_1
    maxq = 15

    weight = v.resize(v.shape[0], v.shape[1] // group_size, group_size)

    tmp = np.zeros([weight.shape[0], weight.shape[1]])
    wmin = np.minimum(weight.min(2)[0], tmp)
    wmax = np.maximum(weight.max(2)[0], tmp)

    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1

    scales = (wmax - wmin) / maxq
    zero = np.atleast_3d(np.round(-wmin / scales))
    scales = np.atleast_3d(scales)
    zero_scales = np.atleast_3d(wmin)

    qq = np.asarray((weight - zero_scales) / scales + 0.5, dtype=np.int16)
    print(f"qq.max = {qq.max()} min = {qq.min()}")
    qweight = np.clip(qq, 0, maxq)

    addends = zero_scales

    addends_view = np.asarray(addends, dtype=np.float16).view(dtype=np.int16)
    scales_view = np.asarray(scales, dtype=np.float16).view(dtype=np.int16)

    # Split into groups of 8 columns (i.e. 64 columns of quantized data):
    # TODO: Only support act-order=false
    grouped = to_ggml_int16(qweight)

    blob = np.concatenate([scales_view, addends_view, grouped], axis=2, casting='no')

    # header
    write_header(fout, shape, dst_name, ftype)

    # data
    # v.numpy().tofile(fout)
    blob.tofile(fout)


def expandToInt4(qweight):
    eweight = qweight.repeat(8, axis=2)
    eweight = eweight.astype(np.uint32)
    for i in range(0, eweight.shape[2]):
        offset = i % (32 // 4) * 4
        eweight[:, :, i] = eweight[:, :, i] >> offset & (2 ** 4 - 1)
    return eweight


def to_ggml_int16(eweight):
    qweight = np.zeros((eweight.shape[0], eweight.shape[1], eweight.shape[2] // 4), dtype=np.uint16)
    eweight = np.asarray(eweight, dtype=np.uint16)
    for i in range(0, qweight.shape[2]):
        qweight[:, :, i] = eweight[:, :, i * 2 + 0]
        qweight[:, :, i] |= eweight[:, :, i * 2 + 32] << 1 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 1] << 2 * 4
        qweight[:, :, i] |= eweight[:, :, i * 2 + 33] << 3 * 4
    return qweight.astype(np.int16)


def qzeros_to_zeros(qzeros, bits=4):
    zeros = np.zeros((qzeros.shape[0], qzeros.shape[1] * (32 // bits)), dtype=np.float32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + (32 // bits)):
            zeros[:, j] = (qzeros[:, col] >> (bits * (j - i)) & (2 ** bits - 1)) + 1
        i += 32 // bits
        col += 1
    return zeros


def convert_q4(src_name, dst_name, model, fout, n_head, permute=False):
    qzeros = model[f"{src_name}.qzeros"].numpy()
    zeros = qzeros_to_zeros(qzeros).T
    scales = model[f"{src_name}.scales"].numpy().T
    qweight = model[f"{src_name}.qweight"].numpy().T  # transpose

    # Q4_1 does not support bias; good thing the bias is always all zeros.
    # Act-order is not supported.
    if f"{src_name}.g_idx" in model:
        g_idx = model[f"{src_name}.g_idx"].numpy()
        invalidInputError(np.all(g_idx[:-1] <= g_idx[1:]),
                          "Act-order is not supported, please use a no act-order model.")
    ftype = 3  # Q4_1

    # Each int32 item is actually 8 int4 items packed together, and it's transposed.
    shape = (qweight.shape[0], qweight.shape[1] * 8)

    print("Processing Q4 variable: " + src_name + " with shape: ", shape)

    # The output format has the int4 weights in groups of 32 rather than 8.
    # It looks like this:
    # For each row:
    #   For each group of 32 columns:
    #     - addend (float32, 4 bytes)
    #     - scale (float32, 4 bytes)
    #     - weights (int4 * 32, 16 bytes)
    # Note that in the input, the scales and addends are shared between all
    # the columns in a row, so we end up wasting quite a bit of memory with
    # repeated scales and addends.

    addends = -zeros * scales  # flip sign

    # Since the output format is mixed between integers and floats, we have
    # to hackily view the floats as int32s just so numpy will let us
    # concatenate them.
    addends_view = np.asarray(addends, dtype=np.float16).view(dtype=np.int16)
    scales_view = np.asarray(scales, dtype=np.float16).view(dtype=np.int16)

    # Split into groups of 8 columns (i.e. 64 columns of quantized data):
    # TODO: Only support act-order=false
    expanded = expandToInt4(qweight.reshape([qweight.shape[0], qweight.shape[1] // 8, 8]))
    if permute:
        qkv = torch.tensor(np.asarray(expanded, dtype=np.float32))\
              .reshape([expanded.shape[0], expanded.shape[1] * expanded.shape[2]])
        q, k, v = qkv.reshape(n_head, 3, -1).unbind(1)
        expanded = np.asarray(torch.cat([q, k, v], dim=0).reshape(expanded.shape).numpy(), expanded.dtype)

    grouped = to_ggml_int16(expanded)

    # Repeat addends and scales:
    if addends_view.shape[1] == grouped.shape[1]:
        addends_rep = np.atleast_3d(addends_view)
        scales_rep = np.atleast_3d(scales_view)
    else:
        addends_rep = np.atleast_3d(addends_view) \
            .repeat(grouped.shape[1] // addends_view.shape[1], axis=1)
        scales_rep = np.atleast_3d(scales_view) \
            .repeat(grouped.shape[1] // scales_view.shape[1], axis=1)

    blob = np.concatenate([scales_rep, addends_rep, grouped], axis=2, casting='no')

    # if permute:
    #     # Permute some rows to undo the permutation done by convert_llama_weights_to_hf.py.
    #     # This can be done after the above conversion because it doesn't affect column order/layout.
    #     blob = (blob.reshape(n_head, 2, shape[0] // n_head // 2, *blob.shape[1:])
    #             .swapaxes(1, 2)
    #             .reshape(blob.shape))

    # header
    write_header(fout, shape, dst_name, ftype)  # ftype = Q4_1

    # data
    blob.tofile(fout)


def find_quantized_model_file(model_path):
    model_path = Path(model_path)
    for ext in ['.safetensors', '.pt']:
        found = list(model_path.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) != 1:
                warnings.warn(f'Detected {len(found)} {ext} model, use the first one {found[0]}.')
            print(f"Detected model file {found[0]}")
            return str(found[0])


def convert_gptq2ggml(model_path, output_path, tokenizer_path=None):
    input_path = find_quantized_model_file(model_path)

    # To debug f16_to_q4
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model = model.state_dict()

    # if input_path.endswith('pt'):
    #     model = torch.load(input_path, map_location="cpu")
    # elif input_path.endswith('safetensors'):
    #     from safetensors.torch import load_file
    #     model = load_file(input_path)
    # else:
    #     invalidInputError(False, "unknown input model path, only support .safetensors or .pt file.")

    embedding_layer = model[next(iter(model))]
    n_vocab, n_embd = embedding_layer.shape
    # layer_re = r'model\.layers\.([0-9]+)'
    layer_re = r'transformer\.h\.([0-9]+)'
    # n_vocab, n_embd = model['transformer.word_embeddings.weight'].shape
    # layer_re = r'transformer\.h\.([0-9]+)'
    n_layer = 1 + max(int(re.match(layer_re, name).group(1)) for name in model
                      if re.match(layer_re, name))

    # hardcoded:
    # n_mult = 256
    # n_head = {32: 32, 40: 40, 60: 52, 80: 64}[n_layer]
    n_mult = 1
    n_head = 32

    if not tokenizer_path:
        tokenizer_path = os.path.join(model_path, "tokenizer.model")

    if os.path.isfile(tokenizer_path):
        tokenizer = SentencePieceProcessor(tokenizer_path)
        vocab_size = tokenizer.vocab_size()
        autoToken = False
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab_size = tokenizer.vocab_size
        dot_token = tokenizer.encode(".")[0]
        autoToken = True

    # TODO: Support AutoTokenizer

    invalidInputError(vocab_size <= n_vocab, "vocab size not match.")
    if vocab_size < n_vocab:
        print(f"tokenizer's vocab size {vocab_size} is the smaller than embedding size {n_vocab}, "
              f"will delete the useless embedding layer.")
        model["transformer.word_embeddings.weight"] = model["transformer.word_embeddings.weight"][0:vocab_size, :]
        model["lm_head.weight"] = model["lm_head.weight"][0:vocab_size, :]

    fout = open(output_path, "wb")

    # fout.write(b"ggjt"[::-1])  # magic: ggmf in hex
    # fout.write(b"ggjt"[::-1])  # magic: ggmf in hex
    values = [0x67676d6c,  # file version
              vocab_size,
              n_embd,
              n_mult,
              n_head,
              n_layer,
              3]
    # n_embd // n_head,  # rot (obsolete)
    fout.write(struct.pack("i" * len(values), *values))

    for i in range(vocab_size):
        if autoToken:
            text = tokenizer.decode([i]).encode('utf-8')
        elif tokenizer.is_unknown(i):
            text = " \u2047 ".encode("utf-8")
        elif tokenizer.is_control(i):
            text = b""
        elif tokenizer.is_byte(i):
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                print(f"Invalid token: {piece}")
                sys.exit(1)
            byte_value = int(piece[3:-1], 16)
            text = struct.pack("B", byte_value)
        else:
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        if not autoToken:
            fout.write(struct.pack("f", tokenizer.get_score(i)))

    convert_fp_to_q4("transformer.word_embeddings.weight", "tok_embeddings.weight", model, fout)
    convert_non_q4("transformer.word_embeddings_layernorm.weight", "norm.weight", model, fout)
    convert_non_q4("transformer.word_embeddings_layernorm.bias", "norm.bias", model, fout)

    # for i in range(n_layer):
    #     convert_non_q4(f"transformer.h.{i}.input_layernorm.weight",
    #                    f"layers.{i}.attention_norm.weight", model, fout)
    #     convert_non_q4(f"transformer.h.{i}.input_layernorm.bias",
    #                    f"layers.{i}.attention_norm.bias", model, fout)
    #     convert_q4(f"transformer.h.{i}.self_attention.query_key_value",
    #                f"layers.{i}.attention.query_key_value.weight", model, fout, n_head, permute=True)
    #     # convert_q4(f"transformer.h.{i}.self_attention.query_key_value",
    #     #            f"layers.{i}.attention.query_key_value.weight", model, fout, n_head)
    #     convert_non_q4(f"transformer.h.{i}.self_attention.query_key_value.bias",
    #                    f"layers.{i}.attention.query_key_value.bias", model, fout)
    #     convert_q4(f"transformer.h.{i}.self_attention.dense",
    #                f"layers.{i}.attention.wo.weight", model, fout, n_head)
    #     convert_non_q4(f"transformer.h.{i}.self_attention.dense.bias",
    #                    f"layers.{i}.attention.wo.bias", model, fout)
    #     convert_non_q4(f"transformer.h.{i}.post_attention_layernorm.weight",
    #                    f"layers.{i}.ffn_norm.weight", model, fout)
    #     convert_non_q4(f"transformer.h.{i}.post_attention_layernorm.bias",
    #                    f"layers.{i}.ffn_norm.bias", model, fout)
    #     convert_q4(f"transformer.h.{i}.mlp.dense_h_to_4h",
    #                f"layers.{i}.feed_forward.w1.weight", model, fout, n_head)
    #     convert_non_q4(f"transformer.h.{i}.mlp.dense_h_to_4h.bias",
    #                    f"layers.{i}.feed_forward.w1.bias", model, fout)
    #     convert_q4(f"transformer.h.{i}.mlp.dense_4h_to_h",
    #                f"layers.{i}.feed_forward.w2.weight", model, fout, n_head)
    #     convert_non_q4(f"transformer.h.{i}.mlp.dense_4h_to_h.bias",
    #                    f"layers.{i}.feed_forward.w2.bias", model, fout)
    for i in range(n_layer):
        convert_non_q4(f"transformer.h.{i}.input_layernorm.weight",
                       f"layers.{i}.attention_norm.weight", model, fout)
        convert_non_q4(f"transformer.h.{i}.input_layernorm.bias",
                       f"layers.{i}.attention_norm.bias", model, fout)
        convert_fp_to_q4(f"transformer.h.{i}.self_attention.query_key_value.weight",
                   f"layers.{i}.attention.query_key_value.weight", model, fout)
        # convert_q4(f"transformer.h.{i}.self_attention.query_key_value",
        #            f"layers.{i}.attention.query_key_value.weight", model, fout, n_head)
        convert_non_q4(f"transformer.h.{i}.self_attention.query_key_value.bias",
                       f"layers.{i}.attention.query_key_value.bias", model, fout)
        convert_fp_to_q4(f"transformer.h.{i}.self_attention.dense.weight",
                         f"layers.{i}.attention.wo.weight", model, fout)
        convert_non_q4(f"transformer.h.{i}.self_attention.dense.bias",
                       f"layers.{i}.attention.wo.bias", model, fout)
        convert_non_q4(f"transformer.h.{i}.post_attention_layernorm.weight",
                       f"layers.{i}.ffn_norm.weight", model, fout)
        convert_non_q4(f"transformer.h.{i}.post_attention_layernorm.bias",
                       f"layers.{i}.ffn_norm.bias", model, fout)
        convert_fp_to_q4(f"transformer.h.{i}.mlp.dense_h_to_4h.weight",
                   f"layers.{i}.feed_forward.w1.weight", model, fout)
        convert_non_q4(f"transformer.h.{i}.mlp.dense_h_to_4h.bias",
                       f"layers.{i}.feed_forward.w1.bias", model, fout)
        convert_fp_to_q4(f"transformer.h.{i}.mlp.dense_4h_to_h.weight",
                   f"layers.{i}.feed_forward.w2.weight", model, fout)
        convert_non_q4(f"transformer.h.{i}.mlp.dense_4h_to_h.bias",
                       f"layers.{i}.feed_forward.w2.bias", model, fout)
    convert_non_q4("transformer.ln_f.weight", "output_norm.weight", model, fout)
    convert_non_q4("transformer.ln_f.bias", "output_norm.bias", model, fout)
    convert_fp_to_q4("lm_head.weight", "output.weight", model, fout)
    fout.close()
    print("Done. Output file: " + output_path)
    print("")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: convert-gptq-to-ggml.py llamaXXb-4bit.pt tokenizer.model out.bin\n")
        sys.exit(1)

    fname_model = sys.argv[1]
    fname_tokenizer = sys.argv[2]
    out_path = sys.argv[3]
    convert_gptq2ggml(fname_model, out_path, tokenizer_path=fname_tokenizer)
