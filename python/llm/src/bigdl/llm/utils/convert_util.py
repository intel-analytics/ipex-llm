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
# ===========================================================================
#
# This file is adapted from
# https://github.com/ggerganov/llama.cpp/blob/master/convert.py
# https://github.com/togethercomputer/redpajama.cpp/blob/master/examples/redpajama/scripts/convert_gptneox_to_ggml.py
# https://github.com/NouamaneTazi/bloomz.cpp/blob/patch/convert-hf-to-ggml.py
#
# MIT License
#
# Copyright (c) 2023 Georgi Gerganov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import concurrent.futures
import copy
import enum
import faulthandler
import functools
import io
import itertools
import json
import math
import mmap
import pickle
import re
import signal
import struct
import sys
import zipfile
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Literal, Optional, Sequence, Tuple, TypeVar, Union)
import numpy as np
from sentencepiece import SentencePieceProcessor
from bigdl.llm.utils.common import invalidInputError
import os
from pathlib import Path

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray = np.ndarray[Any, Any]


__all__ = ['Params',
           'OutputFile',
           'load_some_model',
           'do_necessary_conversions',
           'pick_output_type',
           'convert_to_output_type',
           'load_vocab',
           'default_outfile',
           '_convert_gptneox_hf_to_ggml',
           '_convert_bloom_hf_to_ggml',
           '_convert_starcoder_hf_to_ggml',
           '_convert_chatglm_hf_to_ggml']


@dataclass(frozen=True)
class UnquantizedDataType:
    name: str


DT_F16 = UnquantizedDataType('F16')
DT_F32 = UnquantizedDataType('F32')
DT_I32 = UnquantizedDataType('I32')
DT_BF16 = UnquantizedDataType('BF16')


@dataclass(frozen=True)
class QuantizedDataType:
    groupsize: int
    have_addends: bool
    have_g_idx: bool


DT_Q4_0 = QuantizedDataType(groupsize=32, have_addends=False, have_g_idx=False)
DT_Q4_1 = QuantizedDataType(groupsize=32, have_addends=True, have_g_idx=False)

DataType = Union[UnquantizedDataType, QuantizedDataType]

DATA_TYPE_TO_FTYPE = Dict[DataType, int]
DATA_TYPE_TO_FTYPE = {DT_F32: 0,
                      DT_F16: 1,
                      DT_Q4_0: 2,
                      DT_Q4_1: 3}

FTYPE_TO_DATA_TYPE = Dict[int, DataType]
FTYPE_TO_DATA_TYPE = {ftype: dtype for (dtype, ftype) in DATA_TYPE_TO_FTYPE.items()}

DATA_TYPE_TO_NUMPY = Dict[DataType, 'np.dtype[Any]']
DATA_TYPE_TO_NUMPY = {DT_BF16: np.dtype(np.uint16),
                      DT_F16: np.dtype(np.float16),
                      DT_F32: np.dtype(np.float32),
                      DT_I32: np.dtype(np.int32)}

NUMPY_TYPE_TO_DATA_TYPE = Dict['np.dtype[Any]', DataType]
NUMPY_TYPE_TO_DATA_TYPE = {dtype: data_type for (data_type, dtype) in DATA_TYPE_TO_NUMPY.items()}


class GGMLFileType(enum.Enum):
    AllF32 = 0
    MostlyF16 = 1  # except 1d tensors
    MostlyQ4_0 = 2  # except 1d tensors
    MostlyQ4_1 = 3  # except 1d tensors
    PerLayerIsQ4_1 = 4  # but tok_embeddings.weight and output.weight are F16

    def type_for_tensor(self, name: str, tensor: 'LazyTensor') -> DataType:
        if len(tensor.shape) == 1:
            # 1D tensors are always F32.
            return DT_F32
        elif self == GGMLFileType.AllF32:
            return DT_F32
        elif self == GGMLFileType.MostlyF16:
            return DT_F16
        elif self == GGMLFileType.MostlyQ4_0:
            return DT_Q4_0
        elif self == GGMLFileType.MostlyQ4_1:
            return DT_Q4_1
        elif self == GGMLFileType.PerLayerIsQ4_1:
            if name in ('output.weight', 'tok_embeddings.weight'):
                return DT_F16
            else:
                return DT_Q4_1
        else:
            invalidInputError(False, 'There exists ValueError.')


def make_tensors_list() -> List[str]:
    ret = [
        'tok_embeddings.weight',
        'norm.weight',
        'output.weight',
    ]
    for i in range(80):  # maximum number of layer
        ret += [
            f'layers.{i}.attention.wq.weight',
            f'layers.{i}.attention.wk.weight',
            f'layers.{i}.attention.wv.weight',
            f'layers.{i}.attention.wo.weight',
            f'layers.{i}.attention_norm.weight',
            f'layers.{i}.feed_forward.w1.weight',
            f'layers.{i}.feed_forward.w2.weight',
            f'layers.{i}.feed_forward.w3.weight',
            f'layers.{i}.atttention_norm.weight',
            f'layers.{i}.ffn_norm.weight',
        ]
    return ret


TENSORS_LIST = make_tensors_list()
TENSORS_SET = set(TENSORS_LIST)


def find_n_mult(n_ff: int, n_embd: int) -> int:
    # hardcoded magic range
    for n_mult in range(8192, 1, -1):
        calc_ff = (((8*n_embd) // 3 + n_mult - 1) // n_mult)*n_mult
        if calc_ff == n_ff:
            return n_mult
    invalidInputError(False,
                      f"Failed to find n_mult for (n_ff={n_ff}, n_embd={n_embd}).")


@dataclass
class Params:
    n_vocab:   int
    n_embd:    int
    n_mult:    int
    n_head:    int
    n_layer:   int
    n_kv_head: Optional[int]  # This parameter is only used for Llama 2

    @staticmethod
    def guessed(model: 'LazyModel') -> 'Params':
        # try transformer naming first
        if "model.embed_tokens.weight" in model:
            n_vocab, n_embd = model["model.embed_tokens.weight"].shape
        else:
            n_vocab, n_embd = model["tok_embeddings.weight"].shape

        # try transformer naming first
        if "model.layers.0.self_attn.q_proj.weight" in model:
            n_layer = next(i for i in itertools.count()
                           if f"model.layers.{i}.self_attn.q_proj.weight" not in model)
        elif "model.layers.0.self_attn.W_pack.weight" in model:   # next: try baichuan naming
            n_layer = next(i for i in itertools.count()
                           if f"model.layers.{i}.self_attn.W_pack.weight" not in model)
        else:
            n_layer = next(i for i in itertools.count()
                           if f"layers.{i}.attention.wq.weight" not in model)

        if n_layer < 1:
            invalidInputError(False, "Failed to guess 'n_layer'. This model is unknown or "
                                     "unsupported.\nSuggestion: provide 'config.json' of the "
                                     "model in the same directory containing model files.")

        n_head = n_embd // 128  # guessed

        return Params(
            n_vocab=n_vocab,
            n_embd=n_embd,
            n_mult=256,
            n_head=n_head,
            n_layer=n_layer,
            n_kv_head=None,
        )

    @staticmethod
    def loadHFTransformerJson(model: 'LazyModel', config_path: 'Path') -> 'Params':
        config = json.load(open(config_path))

        n_vocab = config["vocab_size"]
        n_embd = config["hidden_size"]
        n_head = config["num_attention_heads"]
        n_layer = config["num_hidden_layers"]
        n_ff = config["intermediate_size"]
        n_kv_head = config.get("num_key_value_heads")

        n_mult = find_n_mult(n_ff, n_embd)

        return Params(
            n_vocab=n_vocab,
            n_embd=n_embd,
            n_mult=n_mult,
            n_head=n_head,
            n_layer=n_layer,
            n_kv_head=n_kv_head,
        )

    # LLaMA v2 70B params.json
    # {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8,
    # "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1}
    @staticmethod
    def loadOriginalParamsJson(model: 'LazyModel', config_path: 'Path') -> 'Params':
        config = json.load(open(config_path))

        n_vocab = config["vocab_size"]
        n_embd = config["dim"]
        n_head = config["n_heads"]
        n_layer = config["n_layers"]
        n_mult = config["multiple_of"]

        if n_vocab == -1:
            n_vocab = model["tok_embeddings.weight"].shape[0]

        return Params(
            n_vocab=n_vocab,
            n_embd=n_embd,
            n_mult=n_mult,
            n_head=n_head,
            n_layer=n_layer,
            n_kv_head=None,
        )

    @staticmethod
    def load(model_plus: 'ModelPlus') -> 'Params':
        hf_config_path = model_plus.paths[0].parent / "config.json"
        orig_config_path = model_plus.paths[0].parent / "params.json"

        if hf_config_path.exists():
            params = Params.loadHFTransformerJson(model_plus.model, hf_config_path)
        elif orig_config_path.exists():
            params = Params.loadOriginalParamsJson(model_plus.model, orig_config_path)
        else:
            params = Params.guessed(model_plus.model)

        print(f'params: n_vocab:{params.n_vocab} n_embd:{params.n_embd}'
              f'n_mult:{params.n_mult} n_head:{params.n_head} n_layer:{params.n_layer}')
        return params


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Optional[Path],
                 vocabtype: Optional[str]) -> None:
        self.vocabtype = vocabtype
        if self.vocabtype == "bpe":
            self.sentencepiece_tokenizer = json.loads(open(str(fname_tokenizer)).read())
        else:
            self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens = Dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens))
        else:
            added_tokens = {}
        if self.vocabtype == "bpe":
            vocab_size: int = len(self.sentencepiece_tokenizer)
        else:
            vocab_size: int = self.sentencepiece_tokenizer.vocab_size()
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        invalidInputError(expected_ids == actual_ids,
                          "Expected added token IDs to be sequential and start "
                          f"at {len(added_tokens)}; got {actual_ids}")
        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base = vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[Tuple[bytes, float]]:
        tokenizer = self.sentencepiece_tokenizer
        if self.vocabtype == "bpe":
            from transformers.models.gpt2 import tokenization_gpt2
            byte_encoder = tokenization_gpt2.bytes_to_unicode()
            byte_decoder = {v: k for k, v in byte_encoder.items()}
            for i, item in enumerate(tokenizer):
                text: bytes
                text = b''.join([x.to_bytes(1, byteorder='big') for x in [byte_decoder[y]
                                 for y in item]])
                score: float = -i
                yield text, score
        else:
            for i in range(tokenizer.vocab_size()):
                text: bytes
                if tokenizer.is_unknown(i):
                    text = " \u2047 ".encode("utf-8")
                elif tokenizer.is_control(i):
                    text = b""
                elif tokenizer.is_byte(i):
                    piece = tokenizer.id_to_piece(i)
                    if len(piece) != 6:
                        invalidInputError(False, f"Invalid token: {piece}")
                    byte_value = int(piece[3:-1], 16)
                    text = struct.pack("B", byte_value)
                else:
                    text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
                score: float = tokenizer.get_score(i)
                yield text, score

    def added_tokens(self) -> Iterable[Tuple[bytes, float]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens" + \
            f"and {len(self.added_tokens_list)} added tokens>"


class GGMLVocab:
    def __init__(self, tokens: List[Tuple[bytes, float]]):
        self.tokens = tokens
        self.vocab_size = len(tokens)

    def all_tokens(self) -> Iterable[Tuple[bytes, float]]:
        return self.tokens

    def __repr__(self) -> str:
        return f"<GGMLVocab with {self.vocab_size} tokens>"


Vocab = Union[SentencePieceVocab, GGMLVocab]


def permute(weights: NDArray, n_head: int, n_kv_head: Optional[int] = None) -> NDArray:
    if n_kv_head is not None and n_head != n_kv_head:
        n_head //= n_kv_head
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(weights.shape))


def dequantize_q4(qvalues_pack32: NDArray, scales: NDArray,
                  addends: Optional[NDArray], g_idx: Optional[NDArray]) -> NDArray:
    # First reinterpret each row from a list of int32s containing 8 values each
    # to a list of uint8s containing 2 values each.
    qvalues_pack8 = qvalues_pack32.view(np.uint8)

    # Then split out the two values per int8 (which requires an actual
    # conversion because numpy doesn't natively support int4s).
    qvalues = np.zeros([qvalues_pack8.shape[0], qvalues_pack8.shape[1] * 2], dtype=np.uint8)
    qvalues[:, 0::2] = qvalues_pack8 & 0xf
    qvalues[:, 1::2] = qvalues_pack8 >> 4

    invalidInputError(addends is None or addends.shape == scales.shape,
                      "Fail during dequantization because addends and scales dismatch.")
    invalidInputError(qvalues.shape[0] == scales.shape[0] and
                      qvalues.shape[1] % scales.shape[1] == 0,
                      "Fail during dequantization because qvalues and scales dismatch.")
    if g_idx is None:
        repeat_count = qvalues.shape[1] // scales.shape[1]
        scales = scales[:, :, np.newaxis]
        if addends is not None:
            addends = addends[:, :, np.newaxis]
        # Reshape so that the below computation broadcasts over scales and addends:
        qvalues.shape = (qvalues.shape[0], scales.shape[1], int(repeat_count))
    else:
        # In this case the scale and addend is selected for each column by g_idx:
        invalidInputError(addends is not None,
                          "The addend is selected for each column by g_idx, but got None.")
        scales = scales[:, g_idx]
        addends = addends[:, g_idx]
    if addends is None:
        # Q4_0
        qvalues = qvalues.view(np.int8)
        qvalues -= 8
    # And do the actual 'value = scale * qvalue + addend' computation.
    values = scales * qvalues
    if addends is not None:
        values += addends
    if g_idx is None:
        values.shape = (values.shape[0], values.shape[1] * values.shape[2])
    return values


class Tensor(metaclass=ABCMeta):
    data_type: DataType

    @abstractmethod
    def astype(self, data_type: DataType) -> 'Tensor':
        pass

    @abstractmethod
    def permute(self, n_head: int, n_kv_head: Optional[int] = None) -> 'Tensor':
        pass

    @abstractmethod
    def permute_part(self, n_part: int, n_head: int) -> 'UnquantizedTensor':
        pass

    @abstractmethod
    def part(self, n_part: int) -> 'UnquantizedTensor':
        pass

    @abstractmethod
    def to_ggml(self) -> 'GGMLCompatibleTensor':
        pass


def bf16_to_fp32(bf16_arr: np.ndarray) -> np.ndarray:
    invalidInputError(bf16_arr.dtype == np.uint16,
                      f"Input array should be of dtype uint16, but got {bf16_arr.dtype}.")
    fp32_arr = bf16_arr.astype(np.uint32) << 16
    return fp32_arr.view(np.float32)


class UnquantizedTensor(Tensor):
    def __init__(self, ndarray: NDArray) -> None:
        self.ndarray = ndarray
        self.data_type = NUMPY_TYPE_TO_DATA_TYPE[ndarray.dtype]

    def astype(self, data_type: DataType) -> Tensor:
        dtype = DATA_TYPE_TO_NUMPY[data_type]
        if self.data_type == DT_BF16:
            self.ndarray = bf16_to_fp32(self.ndarray)
        return UnquantizedTensor(self.ndarray.astype(dtype))

    def to_ggml(self) -> 'UnquantizedTensor':
        return self

    def permute_part(self, n_part: int, n_head: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(permute(self.ndarray[r * n_part: r * n_part + r, ...], n_head))

    def part(self, n_part: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(self.ndarray[r * n_part: r * n_part + r, ...])

    def permute(self, n_head: int, n_kv_head: Optional[int] = None) -> 'UnquantizedTensor':
        return UnquantizedTensor(permute(self.ndarray, n_head, n_kv_head))


def load_unquantized(lazy_tensor: 'LazyTensor', expected_dtype: Any = None,
                     convert: bool = False) -> NDArray:
    tensor = lazy_tensor.load()

    # double-check:
    actual_shape = list(tensor.ndarray.shape)
    if expected_dtype is not None and expected_dtype != tensor.ndarray.dtype:
        if convert:
            tensor.ndarray = tensor.ndarray.astype(expected_dtype)
        else:
            invalidInputError(False,
                              f'Expected this tensor to have dtype {expected_dtype},'
                              f' but got {tensor.ndarray.dtype}.')

    return tensor.ndarray


class GGMLQuantizedTensor(Tensor):
    data_type: QuantizedDataType

    def __init__(self, ndarray: NDArray, shape: List[int], data_type: DataType) -> None:
        rows, columns = shape
        invalidInputError(columns % data_type.groupsize == 0,
                          "Initialization of GGMLQuantizedTensor failed.")
        words_in_block = 6 if data_type == DT_Q4_1 else 5
        self.ndarray = ndarray.view(dtype=np.uint32) \
                              .reshape((rows, columns // data_type.groupsize, words_in_block))
        self.shape = shape[:]
        self.data_type = data_type

    def astype(self, data_type: DataType) -> Tensor:
        if data_type == self.data_type:
            return self
        scales = self.ndarray[:, :, 0].view(np.float32)
        if self.data_type.have_addends:
            addends = self.ndarray[:, :, 1].view(np.float32)
        else:
            addends = None
        qweights = self.ndarray[:, :, -4:].reshape([self.shape[0], self.shape[1] // 8])

        dq = dequantize_q4(qweights, scales, addends, g_idx=None)
        return UnquantizedTensor(dq).astype(data_type)

    def to_ggml(self) -> 'GGMLQuantizedTensor':
        return self

    def permute(self, n_head: int, n_kv_head: Optional[int] = None) -> 'GGMLQuantizedTensor':
        return GGMLQuantizedTensor(permute(self.ndarray, n_head, n_kv_head),
                                   self.shape, self.data_type)

    def permute_part(self, n_part: int, n_head: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(permute(self.ndarray[r * n_part: r * n_part + r, ...], n_head))

    def part(self, n_part: int) -> 'UnquantizedTensor':
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(self.ndarray[r * n_part: r * n_part + r, ...])


GGMLCompatibleTensor = Union[UnquantizedTensor, GGMLQuantizedTensor]


class DeferredPermutedTensor(Tensor):
    def __init__(self, base: Tensor, n_head: int, n_kv_head: Optional[int] = None) -> None:
        self.base = base
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.data_type = self.base.data_type

    def astype(self, data_type: DataType) -> Tensor:
        return self.base.astype(data_type).permute(self.n_head, self.n_kv_head)

    def to_ggml(self) -> GGMLCompatibleTensor:
        return self.base.to_ggml().permute(self.n_head, self.n_kv_head)

    def permute(self, n_head: int, n_kv_head: Optional[int] = None) -> Tensor:
        invalidInputError(False, "Shouldn't permute twice.")


class GPTQForLLaMaQuantizedTensor(Tensor):
    def __init__(self, model: 'LazyModel', namebase: str) -> None:
        qweight = load_unquantized(model[f"{namebase}.qweight"], np.int32)
        scales = load_unquantized(model[f"{namebase}.scales"], np.float32, convert=True)

        bias = model.get(f"{namebase}.bias")
        if bias is not None:
            invalidInputError(not np.any(load_unquantized(bias)),
                              'Q4_1 does not support bias')

        if f"{namebase}.zeros" in model:
            zeros = load_unquantized(model[f"{namebase}.zeros"], np.float32)
        else:
            qzeros = load_unquantized(model[f"{namebase}.qzeros"], np.int32)
            invalidInputError(qzeros.dtype == np.int32,
                              'Fail to initiate GPTQForLLaMaQuantizedTensor.')
            zeros = dequantize_q4(qzeros, scales, scales, g_idx=None)
            invalidInputError(zeros.dtype == np.float32,
                              'Fail to initiate GPTQForLLaMaQuantizedTensor.')

        invalidInputError(zeros.shape == scales.shape,
                          'Fail to initiate GPTQForLLaMaQuantizedTensor.')

        # Output is transposed compared to the input, and addends have their sign flipped.
        # Scales and zeros similarly must be transposed but only for newer
        # versions of GPTQ-for-LLaMa; the older versions can be identified by
        # having shape (n_embd, 1).
        qweight = qweight.T
        if scales.shape[1] != 1:
            scales = scales.T
            zeros = zeros.T

        # Output also has signs flipped for the addends.
        self.qweight = qweight
        self.scales = scales
        self.addends = -zeros

        self.g_idx = Optional[NDArray]
        if f"{namebase}.g_idx" in model:
            self.g_idx = load_unquantized(model[f"{namebase}.g_idx"], np.int32)
            invalidInputError(self.g_idx.shape == (qweight.shape[1] * 8,),
                              'Fail to initiate GPTQForLLaMaQuantizedTensor.')
        else:
            self.g_idx = None

        self.shape = [self.qweight.shape[0], self.qweight.shape[1] * 8]
        self.data_type = QuantizedDataType(groupsize=self.groupsize(), have_addends=True,
                                           have_g_idx=(self.g_idx is not None))

    def inspect(self, row: int, col: int) -> None:
        '''For debugging.'''
        qweight = (self.qweight[row, col // 8] >> (4 * (col & 7))) & 0xf
        if self.g_idx is not None:
            group = self.g_idx[col]
        else:
            group = int(col // self.groupsize())
        scale = self.scales[row, group]
        addend = self.addends[row, group]
        with np.printoptions(precision=None, suppress=True):
            print(f'scale:{scale} addend:{addend} qweight:{qweight}')
            print('possible values:', np.arange(16) * scale + addend)
            print('actual value:', qweight * scale + addend)

    def astype(self, data_type: DataType) -> Tensor:
        if isinstance(data_type, QuantizedDataType):
            invalidInputError(self.g_idx is None and data_type.have_addends is True
                              and data_type.have_g_idx is False,
                              "Fail to call `GPTQForLLaMaQuantizedTensor.astype`.")
            return self.regroup(data_type.groupsize)

        dequantized = dequantize_q4(np.ascontiguousarray(self.qweight), self.scales,
                                    self.addends, self.g_idx)
        return UnquantizedTensor(dequantized).astype(data_type)

    def groupsize(self) -> int:
        invalidInputError(self.addends.shape == self.scales.shape and
                          self.shape[1] % self.scales.shape[1] == 0,
                          "Fail to call `GPTQForLLaMaQuantizedTensor.groupsize`.")
        return self.shape[1] // self.scales.shape[1]

    def regroup(self, new_groupsize: int = 32) -> 'GPTQForLLaMaQuantizedTensor':
        # Old versions of GPTQ-for-LLaMa shared scales and addends between all the
        # columns in a row.  Newer versions share them between every set of N
        # columns in a row, where N is the `groupsize` parameter, usually 128.  The
        # output format shares them between every set of 32 columns.  To handle
        # this, duplicate scales and addends for every smaller group.
        # (In the above, 'row' and 'column' are in the sense of the output.)
        invalidInputError(self.g_idx is None,
                          "Fail to call `GPTQForLLaMaQuantizedTensor.regroup`.")
        old_groupsize = self.groupsize()
        invalidInputError(old_groupsize >= new_groupsize and old_groupsize % new_groupsize == 0
                          and old_groupsize,
                          "Fail to call `GPTQForLLaMaQuantizedTensor.regroup`.")
        ret = copy.copy(self)
        ret.addends = self.addends.repeat(old_groupsize // new_groupsize, axis=1)
        ret.scales = self.scales.repeat(old_groupsize // new_groupsize, axis=1)
        ret.data_type = QuantizedDataType(groupsize=new_groupsize, have_addends=True,
                                          have_g_idx=False)
        return ret

    def permute(self, n_head: int, n_kv_head: Optional[int] = None) -> Tensor:
        return DeferredPermutedTensor(self, n_head, n_kv_head)

    def to_ggml(self) -> GGMLQuantizedTensor:
        # The output format looks like this:
        # For each row:
        #   For each group of 32 columns:
        #     - addend (float32, 4 bytes)
        #     - scale (float32, 4 bytes)
        #     - weights (int4 * 32, 16 bytes)

        invalidInputError(self.groupsize() == 32,
                          "Should have been regrouped before converting to ggml.")

        # Since the output format is mixed between integers and floats, we have
        # to hackily view the floats as int32s just so numpy will let us
        # concatenate them.
        addends_view = self.addends.view(dtype=np.int32)[:, :, np.newaxis]
        scales_view = self.scales.view(dtype=np.int32)[:, :, np.newaxis]

        # Split into groups of 4 columns (i.e. 32 columns of quantized data):
        grouped = self.qweight.reshape([self.qweight.shape[0], self.qweight.shape[1] // 4, 4])

        # And concatenate:
        grouped = np.concatenate([scales_view, addends_view, grouped], axis=2, casting='no')

        return GGMLQuantizedTensor(grouped, self.shape, DT_Q4_1)


@dataclass
class LazyTensor:
    _load: Callable[[], Tensor]
    shape: List[int]
    data_type: DataType
    description: str

    def load(self) -> Tensor:
        ret = self._load()
        invalidInputError(ret.data_type == self.data_type and
                          (self.data_type, ret.data_type, self.description),
                          "Fail to load `LazyTensor`.")
        return ret

    def astype(self, data_type: DataType) -> 'LazyTensor':
        self.validate_conversion_to(data_type)

        def load() -> Tensor:
            return self.load().astype(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}')

    def validate_conversion_to(self, data_type: DataType) -> None:
        if data_type == self.data_type:
            return
        if isinstance(data_type, QuantizedDataType):
            invalidInputError(isinstance(self.data_type, QuantizedDataType),
                              "Can't turn an unquantized tensor into"
                              f" a quantized type ({data_type}).")
            if self.data_type.have_g_idx:
                sys.stderr.write(
                    "Error: Input uses the newer GPTQ-for-LLaMa format (using g_idx), "
                    "which is not yet natively supported by GGML. For now "
                    "you can still convert this model by passing `--outtype f16` to dequantize, "
                    "but that will result in a much larger output file for no quality benefit.\n")
                sys.exit(1)
            invalidInputError(not data_type.have_g_idx and self.data_type.have_addends and
                              data_type.have_addends,
                              "Fail to convert to expected data type.")


LazyModel = Dict[str, LazyTensor]


@dataclass
class ModelPlus:
    model: LazyModel
    paths: List[Path]  # Where this was read from.
    format: Literal['ggml', 'torch', 'safetensors']
    vocab: Optional[Vocab]  # For GGML models (which have vocab built in), the vocab.


def merge_sharded(models: List[LazyModel]) -> LazyModel:
    # Original LLaMA models have each file contain one part of each tensor.
    # Use a dict instead of a set to preserve order.
    names = {name: None for model in models for name in model}

    def convert(name: str) -> LazyTensor:
        lazy_tensors = [model[name] for model in models]
        if len(lazy_tensors) == 1:
            # only one file; don't go through this procedure since there might
            # be quantized tensors
            return lazy_tensors[0]
        if len(lazy_tensors[0].shape) == 1:
            # the tensor is just duplicated in every file
            return lazy_tensors[0]
        if name.startswith('tok_embeddings.') or \
           name.endswith('.attention.wo.weight') or \
           name.endswith('.feed_forward.w2.weight'):
            # split by columns
            axis = 1
        else:
            # split by rows
            axis = 0
        concatenated_shape = list(lazy_tensors[0].shape)
        concatenated_shape[axis] = sum(tensor.shape[axis] for tensor in lazy_tensors)

        def load() -> UnquantizedTensor:
            ndarrays = [load_unquantized(tensor) for tensor in lazy_tensors]
            concatenated = np.concatenate(ndarrays, axis=axis)
            return UnquantizedTensor(concatenated)
        description = 'concatenated[[' + '] | ['.join(lt.description for lt in lazy_tensors) + ']]'
        return LazyTensor(load, concatenated_shape, lazy_tensors[0].data_type, description)
    return {name: convert(name) for name in names}


def merge_multifile_models(models_plus: List[ModelPlus]) -> ModelPlus:
    formats = set(mp.format for mp in models_plus)
    invalidInputError(len(formats) == 1, "The input models are different formats.")
    format = formats.pop()
    paths = [path for mp in models_plus for path in mp.paths]
    # Use the first non-None vocab, if any.
    try:
        vocab = next(mp.vocab for mp in models_plus if mp.vocab is not None)
    except StopIteration:
        vocab = None

    if any("model.embed_tokens.weight" in mp.model for mp in models_plus):
        # Transformers models put different tensors in different files, but
        # don't split indivdual tensors between files.
        model = LazyModel
        model = {}
        for mp in models_plus:
            model.update(mp.model)
    else:
        model = merge_sharded([mp.model for mp in models_plus])

    return ModelPlus(model, paths, format, vocab)


def permute_lazy(lazy_tensor: LazyTensor, n_head: int,
                 n_kv_head: Optional[int] = None) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute(n_head, n_kv_head)
    return LazyTensor(load, lazy_tensor.shape, lazy_tensor.data_type,
                      f'permute({n_head}, {n_kv_head}) ' + lazy_tensor.description)


def permute_part_lazy(lazy_tensor: LazyTensor, n_part: int, n_head: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute_part(n_part, n_head)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // 3
    return LazyTensor(load, s, lazy_tensor.data_type,
                      f'permute({n_head}) ' + lazy_tensor.description)


def part_lazy(lazy_tensor: LazyTensor, n_part: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().part(n_part)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // 3
    return LazyTensor(load, s, lazy_tensor.data_type, 'part ' + lazy_tensor.description)


def convert_transformers_to_orig(model: LazyModel, params: Params) -> LazyModel:
    out = {}
    out["tok_embeddings.weight"] = model["model.embed_tokens.weight"]
    out["norm.weight"] = model["model.norm.weight"]
    out["output.weight"] = model["lm_head.weight"]

    for i in itertools.count():
        if f"model.layers.{i}.self_attn.q_proj.weight" in model:
            out[f"layers.{i}.attention.wq.weight"] = \
                permute_lazy(model[f"model.layers.{i}.self_attn.q_proj.weight"], params.n_head)
            out[f"layers.{i}.attention.wk.weight"] = \
                permute_lazy(model[f"model.layers.{i}.self_attn.k_proj.weight"],
                             params.n_head, params.n_kv_head)
            out[f"layers.{i}.attention.wv.weight"] = \
                model[f"model.layers.{i}.self_attn.v_proj.weight"]
        elif f"model.layers.{i}.self_attn.W_pack.weight" in model:
            out[f"layers.{i}.attention.wq.weight"] = \
                permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"],
                                  0, params.n_head)
            out[f"layers.{i}.attention.wk.weight"] = \
                permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"],
                                  1, params.n_head)
            out[f"layers.{i}.attention.wv.weight"] = \
                part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"], 2)
        else:
            break
        out[f"layers.{i}.attention.wo.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]

        out[f"layers.{i}.feed_forward.w1.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
        out[f"layers.{i}.feed_forward.w2.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
        out[f"layers.{i}.feed_forward.w3.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]

        out[f"layers.{i}.attention_norm.weight"] = \
            model[f"model.layers.{i}.input_layernorm.weight"]
        out[f"layers.{i}.ffn_norm.weight"] = \
            model[f"model.layers.{i}.post_attention_layernorm.weight"]
    return out


def handle_quantization(model: LazyModel) -> LazyModel:
    '''Convert a model with entries for 'foo.qweight', 'foo.scales', etc.
    (which resolve to UnquantizedTensors with the raw data) to one with entries
    for 'foo.weight' (which resolve to QuantizedTensors).
    '''
    def convert(name: str) -> Tuple[str, LazyTensor]:
        if name.endswith(".qweight"):
            namebase = name.rsplit('.', 1)[0]
            orig_name = namebase + ".weight"

            lazy_tensor = model[name]
            invalidInputError(len(lazy_tensor.shape) == 2,
                              "Fail to convert a model with entries for 'foo.qweight'.")
            real_shape = [lazy_tensor.shape[1], lazy_tensor.shape[0] * 8]

            # Calculate type.  This replicates the logic in
            # GPTQForLLaMaQuantizedTensor (which is executed when the modelis
            # actually loaded).
            lazy_scales = model[f"{namebase}.scales"]
            scales_width = 1 if lazy_scales.shape[1] == 1 else lazy_scales.shape[0]
            invalidInputError(real_shape[1] % scales_width == 0,
                              "Fail to convert a model with entries for 'foo.qweight'.")
            groupsize = real_shape[1] // scales_width
            have_g_idx = f"{namebase}.g_idx" in model
            data_type = QuantizedDataType(groupsize=groupsize, have_addends=True,
                                          have_g_idx=have_g_idx)

            def load() -> Tensor:
                return GPTQForLLaMaQuantizedTensor(model, namebase)

            return (orig_name, LazyTensor(load, real_shape, data_type, '[quantized]'))
        else:
            return (name, model[name])
    return dict(convert(name) for name in model)

# Functionality that simulates `torch.load` but where individual tensors are
# only loaded into memory on demand, not all at once.
# PyTorch can't do this natively as of time of writing:
# - https://github.com/pytorch/pytorch/issues/64327
# This allows us to de-shard without multiplying RAM usage, and also
# conveniently drops the PyTorch dependency (though we still need numpy).


@dataclass
class LazyStorageKind:
    data_type: DataType


@dataclass
class LazyStorage:
    load: Callable[[int, int], NDArray]
    kind: LazyStorageKind
    description: str


class LazyUnpickler(pickle.Unpickler):
    def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
        super().__init__(fp)
        self.data_base_path = data_base_path
        self.zip_file = zip_file

    def persistent_load(self, pid: Any) -> Any:
        invalidInputError(pid[0] == 'storage' and isinstance(pid[1], LazyStorageKind),
                          "Fail to load.")
        data_type = pid[1].data_type
        filename_stem = pid[2]
        filename = self.data_base_path + '/' + filename_stem
        info = self.zip_file.getinfo(filename)

        def load(offset: int, elm_count: int) -> NDArray:
            dtype = DATA_TYPE_TO_NUMPY.get(data_type)
            invalidInputError(dtype is not None, "Tensor stored in unsupported format.")
            fp = self.zip_file.open(info)
            fp.seek(offset * dtype.itemsize)
            size = elm_count * dtype.itemsize
            data = fp.read(size)
            invalidInputError(len(data) == size, "Fail to load.")
            return np.frombuffer(data, dtype)
        description = f'storage data_type={data_type} path-in-zip={filename}' + \
                      f' path={self.zip_file.filename}'
        return LazyStorage(load=load, kind=pid[1], description=description)

    # @staticmethod
    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,
                               # pyright: ignore[reportSelfClsParameterName]
                               requires_grad: Any, backward_hooks: Any,
                               metadata: Any = None) -> LazyTensor:
        invalidInputError(isinstance(storage, LazyStorage), "Fail to rebuild `LazyTensor`.")

        def load() -> UnquantizedTensor:
            elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size))
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return LazyTensor(load, list(size), storage.kind.data_type, description)

    def rebuild_from_type_v2(func, new_type, args, state):
        return func(*args)

    CLASSES = {
        ('torch._tensor', '_rebuild_from_type_v2'): rebuild_from_type_v2,
        ('torch._utils', '_rebuild_tensor_v2'): lazy_rebuild_tensor_v2,
        ('torch', 'BFloat16Storage'): LazyStorageKind(DT_BF16),
        ('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
        ('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
        ('torch', 'IntStorage'): LazyStorageKind(DT_I32),
        ('torch', 'Tensor'): LazyTensor,
    }

    def find_class(self, module: str, name: str) -> Any:
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]


def lazy_load_torch_file(outer_fp: IO[bytes], path: Path) -> ModelPlus:
    zf = zipfile.ZipFile(outer_fp)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    invalidInputError(len(pickle_paths) == 1 and pickle_paths is not None,
                      "Fail to load torch files.")
    pickle_fp = zf.open(pickle_paths[0], 'r')
    unpickler = LazyUnpickler(pickle_fp,
                              data_base_path=pickle_paths[0][:-4],
                              zip_file=zf)
    model = unpickler.load()
    as_dict = dict(model.items())
    return ModelPlus(model=as_dict, paths=[path], format='torch', vocab=None)


SAFETENSORS_DATA_TYPES = {
    'BF16': DT_BF16,
    'F16': DT_F16,
    'F32': DT_F32,
    'I32': DT_I32,
}


def lazy_load_safetensors_file(fp: IO[bytes], path: Path) -> ModelPlus:
    header_size, = struct.unpack('<Q', fp.read(8))
    header = json.loads(fp.read(header_size))
    # Use mmap for the actual data to avoid race conditions with the file offset.
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
    byte_buf = mapped[8 + header_size:]

    def convert(info: Dict[str, Any]) -> LazyTensor:
        data_type = SAFETENSORS_DATA_TYPES[info['dtype']]
        numpy_dtype = DATA_TYPE_TO_NUMPY[data_type]
        shape = info['shape']
        begin, end = info['data_offsets']
        invalidInputError(0 <= begin <= end <= len(byte_buf) and
                          end - begin == math.prod(shape) * numpy_dtype.itemsize,
                          "Fail to load safetensors files.")
        buf = byte_buf[begin:end]

        def load() -> UnquantizedTensor:
            return UnquantizedTensor(np.frombuffer(buf, dtype=numpy_dtype).reshape(shape))
        description = f'safetensors begin={begin} end={end} type={data_type} path={path}'
        return LazyTensor(load, shape, data_type, description)
    model = {name: convert(info) for (name, info) in header.items()}
    return ModelPlus(model=model, paths=[path], format='safetensors', vocab=None)


def must_read(fp: IO[bytes], length: int) -> bytes:
    ret = fp.read(length)
    invalidInputError(len(ret) >= length, "Unexpectedly reached end of file.")
    return ret


def lazy_load_ggml_file(fp: io.BufferedReader, path: Path) -> ModelPlus:
    magic = must_read(fp, 4)[::-1]
    if magic in (b'ggmf', b'ggjt'):
        version, = struct.unpack("i", must_read(fp, 4))
        invalidInputError(version == 1, "Fail to load ggml files.")
    else:
        invalidInputError(magic == b'ggml', "Fail to load ggml files.")
        version = None
    n_vocab, n_embd, n_mult, n_head, n_layer, rot, file_type = \
        struct.unpack('<7i', must_read(fp, 28))

    tokens = []
    for i in range(n_vocab):
        if i == 32000:
            # HACK: GPT4All messed with the format without changing the magic
            # number.  Specifically, they changed the vocab section to contain
            # `n_vocab - 1` tokens instead of `n_vocab` (i.e. omitting the
            # extra pad token).  Try to detect if we're reading a file like
            # this.
            orig_pos = fp.tell()
            fp.seek(20, io.SEEK_CUR)
            is_gpt4all = fp.read(21) == b'tok_embeddings.weight'
            fp.seek(orig_pos)
            if is_gpt4all:
                break

        length, = struct.unpack("i", must_read(fp, 4))
        text = must_read(fp, length)
        if magic != b'ggml':
            score, = struct.unpack("f", must_read(fp, 4))
            tokens.append((text, score))
    vocab = GGMLVocab(tokens) if magic != b'ggml' else None

    model = {}
    # Use mmap for the actual data to avoid race conditions with the file offset.
    off = fp.raw.tell()
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
    # needed on Windows
    fp.raw.seek(off)

    def read_tensor() -> None:
        # this is a function so that variables captured in `load` don't change
        shape_len, name_len, ftype = struct.unpack("iii", must_read(fp, 12))
        invalidInputError(0 <= shape_len <= 3, "Fail to read tensors.")
        shape = list(struct.unpack(f"{shape_len}i", must_read(fp, 4 * shape_len)))
        shape = shape[::-1]
        name = must_read(fp, name_len).decode('utf-8')
        data_type = FTYPE_TO_DATA_TYPE[ftype]

        if magic == b'ggjt':
            fp.seek((fp.tell() + 31) & -32)

        if data_type == DT_Q4_1:
            # See GPTQForLLaMaQuantizedTensor.ggml_ndarray()
            size = 24 * (shape[1] // 32) * shape[0]
        elif data_type == DT_Q4_0:
            size = 20 * (shape[1] // 32) * shape[0]
        else:
            numpy_dtype = DATA_TYPE_TO_NUMPY[data_type]
            elm_count = math.prod(shape)
            size = elm_count * numpy_dtype.itemsize
        offset = fp.tell()
        buf = mapped[offset:offset+size]
        fp.seek(size, io.SEEK_CUR)

        def load() -> Tensor:
            if isinstance(data_type, QuantizedDataType):
                ndarray = np.frombuffer(buf, dtype=np.uint32)
                return GGMLQuantizedTensor(ndarray, shape, data_type)
            else:
                return UnquantizedTensor(np.frombuffer(buf, dtype=numpy_dtype).reshape(shape))
        description = f'ggml offset={offset} type={data_type} path={path}'
        model[name] = LazyTensor(load, shape, data_type, description)

    while fp.read(1) != b'':
        fp.seek(-1, io.SEEK_CUR)
        read_tensor()

    return ModelPlus(model=model, paths=[path], format='ggml', vocab=vocab)


@functools.lru_cache(maxsize=None)
def lazy_load_file(path: Path) -> ModelPlus:
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        # A zip file, i.e. PyTorch format
        return lazy_load_torch_file(fp, path)
    elif first8[2:4] == b'gg':
        # GGML format
        return lazy_load_ggml_file(fp, path)
    elif struct.unpack('<Q', first8)[0] < 16 * 1024 * 1024:
        # Probably safetensors
        return lazy_load_safetensors_file(fp, path)
    else:
        invalidInputError(False, f"unknown format: {path}.")


In = TypeVar('In')
Out = TypeVar('Out')


def bounded_parallel_map(func: Callable[[In], Out], iterable: Iterable[In],
                         concurrency: int) -> Iterable[Out]:
    '''Parallel map, but with backpressure.  If the caller doesn't call `next`
    fast enough, this will stop calling `func` at some point rather than
    letting results pile up in memory.  Specifically, there is a max of one
    output value buffered per thread.'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        items_rev = list(iterable)[::-1]
        for i in range(min(concurrency, len(items_rev))):
            futures.append(executor.submit(func, items_rev.pop()))
        while futures:
            result = futures.pop(0).result()
            if items_rev:
                futures.append(executor.submit(func, items_rev.pop()))
            yield result


def check_vocab_size(params: Params, vocab: Vocab) -> None:
    if params.n_vocab != vocab.vocab_size:
        # GGMLVocab comes from the same file as the model so shouldn't mismatch:
        invalidInputError(isinstance(vocab, SentencePieceVocab),
                          "Vocab and SentencePieceVocab mismatch.")
        if params.n_vocab == vocab.vocab_size_base:
            print("Ignoring added_tokens.json since model matches vocab size without it.")
            vocab.added_tokens_list = []
            vocab.vocab_size = vocab.vocab_size_base
            return
        msg = f"Vocab size mismatch (model has {params.n_vocab}, but {vocab.fname_tokenizer}"
        if vocab.fname_added_tokens is not None:
            msg += f" combined with {vocab.fname_added_tokens}"
        msg += f" has {vocab.vocab_size})."
        if vocab.vocab_size < params.n_vocab < vocab.vocab_size + 20 and \
           vocab.fname_added_tokens is None:
            msg += " Most likely you are missing added_tokens.json,"
            msg += f" which should be in {vocab.fname_tokenizer.parent})."
        invalidInputError(False, msg)


class OutputFile:
    def __init__(self, fname_out: Path) -> None:
        self.fout = open(fname_out, "wb")

    def write_file_header(self, params: Params, file_type: GGMLFileType) -> None:
        self.fout.write(b"ggjt"[::-1])  # magic
        values = [
            1,  # file version
            params.n_vocab,
            params.n_embd,
            params.n_mult,
            params.n_head,
            params.n_layer,
            params.n_embd // params.n_head,  # rot (obsolete)
            file_type.value,
        ]
        self.fout.write(struct.pack("i" * len(values), *values))

    def write_tensor_header(self, name: str, shape: Sequence[int], data_type: DataType) -> None:
        sname = name.encode('utf-8')
        self.fout.write(struct.pack("iii", len(shape), len(sname), DATA_TYPE_TO_FTYPE[data_type]))
        self.fout.write(struct.pack("i" * len(shape), *shape[::-1]))
        self.fout.write(sname)
        self.fout.seek((self.fout.tell() + 31) & -32)

    def write_vocab(self, vocab: Vocab) -> None:
        for text, score in vocab.all_tokens():
            self.fout.write(struct.pack("i", len(text)))
            self.fout.write(text)
            self.fout.write(struct.pack("f", score))

    @staticmethod
    def write_vocab_only(fname_out: Path, vocab: Vocab) -> None:
        of = OutputFile(fname_out)
        params = Params(n_vocab=vocab.vocab_size, n_embd=0, n_mult=0, n_head=1, n_layer=0)
        of = OutputFile(fname_out)
        of.write_file_header(params, file_type=GGMLFileType.AllF32)
        of.write_vocab(vocab)
        of.fout.close()

    @staticmethod
    def write_all(fname_out: Path, params: Params, file_type: GGMLFileType, model: LazyModel,
                  vocab: Vocab) -> None:
        check_vocab_size(params, vocab)
        of = OutputFile(fname_out)
        of.write_file_header(params, file_type)
        print("Writing vocab...")
        of.write_vocab(vocab)

        def do_item(item: Tuple[str, LazyTensor]) -> NDArray:
            name, lazy_tensor = item
            return lazy_tensor.load().to_ggml().ndarray

        ndarrays = bounded_parallel_map(do_item, model.items(), concurrency=8)
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
            padi = len(str(len(model)))
            print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {name:38s} | size {size:16}"
                  f"| type {lazy_tensor.data_type}")
            of.write_tensor_header(name, lazy_tensor.shape, lazy_tensor.data_type)
            ndarray.tofile(of.fout)
        of.fout.close()


def pick_output_type(model: LazyModel, output_type_str: Optional[str]) -> GGMLFileType:
    wq_type = model["layers.0.attention.wq.weight"].data_type
    if output_type_str == "f32" or (output_type_str is None and wq_type in (DT_F32, DT_BF16)):
        return GGMLFileType.AllF32
    if output_type_str == "f16" or (output_type_str is None and wq_type == DT_F16):
        return GGMLFileType.MostlyF16
    if output_type_str == "q4_1" or (output_type_str is None and
       isinstance(wq_type, QuantizedDataType) and wq_type.have_addends):
        if isinstance(model["output.weight"].data_type, QuantizedDataType):
            return GGMLFileType.MostlyQ4_1
        else:
            return GGMLFileType.PerLayerIsQ4_1
    if output_type_str == "q4_0" or \
       (output_type_str is None and isinstance(wq_type, QuantizedDataType)):
        return GGMLFileType.MostlyQ4_0
    name_to_type = {name: lazy_tensor.data_type for (name, lazy_tensor) in model.items()}
    invalidInputError(False, f"Unexpected combination of types: {name_to_type}.")


def do_necessary_conversions(model: LazyModel, params: Params) -> LazyModel:
    model = handle_quantization(model)

    if "lm_head.weight" in model:
        model = convert_transformers_to_orig(model, params)
    model = filter_and_sort_tensors(model)

    return model


def convert_to_output_type(model: LazyModel, output_type: GGMLFileType) -> LazyModel:
    return {name: tensor.astype(output_type.type_for_tensor(name, tensor))
            for (name, tensor) in model.items()}


def nth_multifile_path(path: Path, n: int) -> Optional[Path]:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the nth path in the model.
    '''
    # Support the following patterns:
    patterns = [
        # - x.00.pth, x.01.pth, etc.
        (r'\.[0-9]{2}\.pth$', f'.{n:02}.pth'),
        # - x-00001-of-00002.bin, x-00002-of-00002.bin, etc.
        (r'-[0-9]{5}-of-(.*)$', fr'-{n:05}-of-\1'),
        # x.bin, x.bin.1, etc.
        (r'(\.[0-9]+)?$', r'\1' if n == 0 else fr'\1.{n}')
    ]
    for regex, replacement in patterns:
        if re.search(regex, path.name):
            new_path = path.with_name(re.sub(regex, replacement, path.name))
            if new_path.exists():
                return new_path
    return None


def find_multifile_paths(path: Path) -> List[Path]:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the whole list of paths in the model.
    '''
    ret = []
    for i in itertools.count():
        nth_path = nth_multifile_path(path, i)
        if nth_path is None:
            break
        ret.append(nth_path)
    if not ret:
        # No matches.  This should only happen if the file was named, e.g.,
        # foo.0, and there was no file named foo.  Oh well, try to process it
        # as a single file.
        return [path]
    return ret


def load_some_model(path: Path) -> ModelPlus:
    '''Load a model of any supported format.'''
    # Be extra-friendly and accept either a file or a directory:
    if path.is_dir():
        globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt", "pytorch_model.bin"]
        files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            # Try GGML too, but with lower priority, since if both a non-GGML
            # model and a GGML model exist in the same directory, we assume the
            # latter was converted from the former.
            files = list(path.glob("ggml-model*.bin*"))
        invalidInputError(files, f"Can't find model in directory {path}.")
        invalidInputError(len(files) == 1,
                          f"Found multiple models in {path}, not sure which to pick: {files}.")
        path = files[0]

    paths = find_multifile_paths(path)
    models_plus = []
    for path in paths:
        print(f"Loading model file {path}")
        models_plus.append(lazy_load_file(path))

    model_plus = merge_multifile_models(models_plus)
    return model_plus


def filter_and_sort_tensors(model: LazyModel) -> LazyModel:
    return {name: model[name] for name in TENSORS_LIST if name in model}


def load_vocab(path: Path, vocabtype: Optional[str]) -> SentencePieceVocab:
    # Be extra-friendly and accept either a file or a directory.  Also, if it's
    # a directory, it might be the model directory, and tokenizer.model might
    # be in the parent of that.
    print(f"vocabtype: {vocabtype}")
    if path.is_dir():
        vocab_file = "tokenizer.model"
        if vocabtype == 'bpe':
            vocab_file = "vocab.json"
        path2 = path / vocab_file
        # Use `.parent` instead of /.. to handle the symlink case better.
        path3 = path.parent / vocab_file
        if path2.exists():
            path = path2
        elif path3.exists():
            path = path3
        else:
            invalidInputError(False,
                              f"Could not find tokenizer.model in {path} or its parent; "
                              "if it's in another directory, pass the directory as --vocab-dir")
    added_tokens_path = path.parent / "added_tokens.json"
    print(f"Loading vocab file {path}")
    return SentencePieceVocab(path, added_tokens_path if added_tokens_path.exists() else None,
                              vocabtype)


def default_outfile(model_paths: List[Path], file_type: GGMLFileType) -> Path:
    namestr = {
        GGMLFileType.AllF32: "f32",
        GGMLFileType.MostlyF16: "f16",
        GGMLFileType.MostlyQ4_0: "q4_0",
        GGMLFileType.MostlyQ4_1: "q4_1",
        GGMLFileType.PerLayerIsQ4_1: "q4_1",
    }[file_type]
    ret = model_paths[0] / f"ggml-model-{namestr}.bin"
    if ret in model_paths:
        sys.stderr.write(
            f"Error: Default output path ({ret}) would overwrite the input. "
            "Please explicitly specify a path using --outfile.\n")
        sys.exit(1)
    return ret


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent
    coverage. This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + \
        list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _convert_gptneox_hf_to_ggml(model_path, outfile_dir, outtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16
                                                 if outtype == "f16" else torch.float32)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hparams = model.config.to_dict()

    filestem = Path(model_path).stem
    fn_out = os.path.join(outfile_dir, f"ggml-{filestem}-{outtype}.bin")
    fout = open(fn_out, "wb")

    ggml_file_magic = 0x67676d66  # 0x67676d6c is unversioned
    ggml_file_version = 0x00000001  # v1

    if outtype == "f16":
        ftype = 1
    else:
        ftype = 0

    hparams["multiple_of"] = 1
    fout.write(struct.pack("i", ggml_file_magic))  # magic: ggmf in hex
    fout.write(struct.pack("i", ggml_file_version))
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["max_position_embeddings"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["num_attention_heads"]))
    fout.write(struct.pack("i", hparams["num_hidden_layers"]))
    fout.write(struct.pack("i", int((hparams["hidden_size"] / hparams["num_attention_heads"])
                                    * hparams["rotary_pct"])))  # rotary_dim
    fout.write(struct.pack("i", int(hparams["use_parallel_residual"])))
    fout.write(struct.pack("i", ftype))

    dot_token = tokenizer.encode(".")[0]
    vocab = tokenizer.vocab
    id2token = {v: k for k, v in vocab.items()}
    for i in range(hparams["vocab_size"]):
        if i in id2token:
            text = id2token[i].encode('utf-8')
        else:
            text = tokenizer.decode([i]).encode('utf-8')
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    list_vars = model.state_dict()

    for name in list_vars.keys():
        if name.startswith('gpt_neox.layers.'):
            if 'attention.masked_bias' in name or 'attention.rotary_emb.inv_freq' in name or \
               'attention.bias' in name:
                continue
        # No gradients for these
        list_vars[name].requires_grad = False
        src = name
        nn = name

        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            data = data.astype(np.float32)

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()


def _convert_bloom_hf_to_ggml(model_path, outfile_dir, outtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    conv_map = {'word_embeddings': 'tok_embeddings',
                'word_embeddings_layernorm': 'norm',
                'input_layernorm': 'attention_norm',
                'self_attention.query_key_value': 'attention.query_key_value',
                'self_attention.dense': 'attention.wo',
                'post_attention_layernorm': 'ffn_norm',
                'mlp.dense_h_to_4h': 'feed_forward.w1',
                'mlp.dense_4h_to_h': 'feed_forward.w2',
                'ln_f': 'output_norm',
                'lm_head': 'output',
                }

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    hparams = config.to_dict()
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                 torch_dtype=torch.float16
                                                 if outtype == "f16" else torch.float32,
                                                 low_cpu_mem_usage=True)

    filestem = Path(model_path).stem
    fn_out = os.path.join(outfile_dir, f"ggml-{filestem}-{outtype}.bin")
    fout = open(fn_out, "wb")

    if outtype == "f16":
        ftype = 1
    else:
        ftype = 0

    hparams["multiple_of"] = 1
    fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    # fout.write(struct.pack("i", hparams["seq_length"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["multiple_of"]))
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", ftype))

    dot_token = tokenizer.encode(".")[0]
    for i in range(hparams["vocab_size"]):
        text = tokenizer.decode([i]).encode('utf-8')
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    list_vars = model.state_dict()
    for name in list_vars.keys():
        src = name
        nn = name
        if name != "lm_head.weight":
            nn = nn.split(".")[1:]
        else:
            nn = nn.split(".")

        if nn[0] == "h":
            nn[0] = "layers"
            mapped = conv_map[".".join(nn[2:-1])]
            name = ".".join(nn[:2] + [mapped] + nn[-1:])
        else:
            mapped = conv_map[".".join(nn[:-1])]
            name = ".".join([mapped] + nn[-1:])

        if "query_key_value" in src:
            q, k, v = list_vars[src].reshape(config.n_head, 3, -1).unbind(1)
            list_vars[src] = torch.cat([q, k, v], dim=0).reshape_as(list_vars[src])

        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            data = data.astype(np.float16)
            ftype_cur = 1

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()


def _convert_starcoder_hf_to_ggml(model_path, outfile_dir, outtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hparams = config.to_dict()
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                 torch_dtype=torch.float16
                                                 if outtype == "f16" else torch.float32,
                                                 # low_cpu_mem_usage=True,
                                                 trust_remote_code=True,
                                                 offload_state_dict=True)

    list_vars = model.state_dict()

    encoder = tokenizer.vocab
    # Add added_tokens (special tokens) to the encoder
    encoder.update(tokenizer.get_added_vocab())

    filestem = Path(model_path).stem
    fn_out = os.path.join(outfile_dir, f"ggml-{filestem}-{outtype}.bin")
    fout = open(fn_out, "wb")

    if outtype == "f16":
        ftype = 1
    else:
        ftype = 0

    fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
    vocab_size = hparams["vocab_size"]
    fout.write(struct.pack("i", vocab_size))
    # fout.write(struct.pack("i", len(encoder)))
    fout.write(struct.pack("i", hparams["n_positions"]))
    fout.write(struct.pack("i", hparams["n_embd"]))
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", ftype))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    fout.write(struct.pack("i", vocab_size))

    counter = 0
    # sort by value
    for key in sorted(encoder, key=encoder.get):
        text = bytearray([byte_decoder[c] for c in key])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    # TODO: Repeat last token until vocab_size
    while counter < vocab_size:
        fout.write(struct.pack("i", len(text)))
        fout.write(text)
        counter += 1

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)

        # rename headers to keep compatibility
        if name == "transformer.ln_f.weight":
            name = "model/ln_f/g"
        elif name == "transformer.ln_f.bias":
            name = "model/ln_f/b"
        elif name == "transformer.wte.weight":
            name = "model/wte"
        elif name == "transformer.wpe.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head"
        elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"transformer.h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"transformer.h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/b"
        else:
            print("Unrecognized variable name. %s", name)

        # we don't need these
        if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
            print("  Skipping variable: " + name)
            continue

        n_dims = len(data.shape)

        ftype_cur = 0
        if ftype == 1:
            if (name == "model/wte" or name == "model/lm_head" or name[-2:] == "/g" or
                    name[-2:] == "/w") and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        "model/h.*/attn/c_attn/w"
        "model/h.*/attn/c_proj/w"
        "model/h.*/mlp/c_fc/w"
        "model/h.*/mlp/c_proj/w"
        if name[-14:] == "/attn/c_attn/w" or name[-14:] == "/attn/c_attn/b":
            print("  Duplicate K,V heads to use MHA instead of MQA")

            embed_dim = hparams["n_embd"]
            head_dim = embed_dim // hparams["n_head"]

            # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
            q, k, v = np.split(data,
                               (hparams["n_head"] * head_dim,
                                (hparams["n_head"] + 1) * head_dim),
                               axis=0)
            # duplicate k, v along the first axis (head_dim, hidden_dim) ->
            # (n_heads * head_dim, hidden_dim)
            if len(k.shape) == 2:
                k = np.tile(k, (hparams["n_head"], 1))
                v = np.tile(v, (hparams["n_head"], 1))
            elif len(k.shape) == 1:
                k = np.tile(k, (hparams["n_head"]))
                v = np.tile(v, (hparams["n_head"]))
            # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) ->
            # (3 * n_heads * head_dim, hidden_dim)
            data = np.concatenate((q, k, v), axis=0)

        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()


def _convert_chatglm_hf_to_ggml(model_path, outfile_dir, outtype):
    filestem = Path(model_path).stem
    outfile = os.path.join(outfile_dir, f"bigdl_llm_chatglm_{outtype}.bin")
    invalidInputError(outtype in ["q4_0", "q4_1"],
                      "For now we only support quantization type 'q4_0' and 'q4_1' "
                      "in chatglm family.")
    from bigdl.llm.utils.convert_chatglm import _convert_chatglm_hf_to_ggml_
    return _convert_chatglm_hf_to_ggml_(model_path,
                                        outfile,
                                        outtype)
