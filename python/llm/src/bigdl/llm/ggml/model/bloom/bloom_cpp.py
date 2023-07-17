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
# https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_cpp.py
#
# MIT License
#
# Copyright (c) 2023 Andrei Betlen
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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

import sys
import os
import ctypes
from typing import List
from ctypes import (
    c_int,
    c_long,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    pointer,
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib
from bigdl.llm.utils import get_shared_lib_info
from bigdl.llm.utils.common import invalidInputError


# Load the library
def _load_shared_library(lib_base_name: str):
    _base_path, _lib_paths = get_shared_lib_info(lib_base_name=lib_base_name)

    if "BLOOM_CPP_LIB" in os.environ:
        lib_base_name = os.environ["BLOOM_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                invalidInputError(False,
                                  f"Failed to load shared library '{_lib_path}': {e}")

    invalidInputError(False, f"Shared library with base name '{lib_base_name}' not found")


# Specify the base name of the shared library to load
_lib_base_name = "bloom"

# Load the library
_lib = _load_shared_library(_lib_base_name)


def c_free(p: c_void_p):
    _lib.c_free(p)


_lib.c_free.argtypes = [c_void_p]
_lib.c_free.restype = None


def bloom_load(fname: bytes, n_ctx: c_int, n_threads: c_int) -> c_void_p:
    return _lib.bloom_load(fname, n_ctx, n_threads)


_lib.bloom_load.argtypes = [c_char_p, c_int, c_int]
_lib.bloom_load.restype = c_void_p


def bloom_free(ctx: c_void_p):
    return _lib.bloom_free(ctx)


_lib.bloom_free.argtypes = [c_void_p]
_lib.bloom_free.restype = None


def bloom_run(ctx: c_void_p,
              seed: c_int,
              n_threads: c_int,
              n_batch: c_int,
              n_predict: c_int,
              match_str: c_bool,
              prompt: bytes,
              buf: bytes) -> c_int:
    return _lib.bloom_run(ctx, seed, n_threads, n_batch, n_predict, match_str, prompt, buf)


_lib.bloom_run.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_bool, c_char_p, c_char_p]
_lib.bloom_run.restype = c_int


def bloom_tokenize(ctx: c_void_p,
                   prompt: bytes,
                   bos: bool = False) -> List[int]:
    n_tokens = c_int(0)
    c_tokens = _lib.tokenize_api(ctx, prompt, bos, pointer(n_tokens))
    tokens = [c_tokens[i] for i in range(0, n_tokens.value)]
    c_free(c_tokens)
    return tokens


_lib.tokenize_api.argtypes = [c_void_p, c_char_p, c_bool, c_void_p]
_lib.tokenize_api.restype = POINTER(c_int)


def bloom_detokenize(ctx: c_void_p,
                     token_id: c_int) -> str:
    c_chars = _lib.detokenize_api(ctx, token_id)
    s = c_chars.decode('utf-8')
    return s


_lib.detokenize_api.argtypes = [c_void_p, c_int]
_lib.detokenize_api.restype = c_char_p


def bloom_eval(ctx: c_void_p,
               input_ids: List[int],
               seed: c_int,
               n_threads: c_int,
               n_batch: c_int) -> List[List[float]]:
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    n_logits = c_long(0)
    c_logits = _lib.eval_api(ctx, c_input_ids, length, seed, n_threads, n_batch, pointer(n_logits))
    n_vocab = n_logits.value // length
    assert(n_vocab * length == n_logits.value)
    logits = [[c_logits[i * n_vocab + j] for j in range(n_vocab)] for i in range(length)]
    # do not free c_logits
    return logits


_lib.eval_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
_lib.eval_api.restype = POINTER(c_float)


def bloom_embed(ctx: c_void_p,
                input_ids: List[int],
                seed: c_int,
                n_threads: c_int,
                n_batch: c_int) -> List[float]:
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    n_embd = c_long(0)
    c_embeddings = _lib.embed_api(ctx, c_input_ids, length, seed, n_threads,
                                  n_batch, pointer(n_embd))
    embeddings = [c_embeddings[i] for i in range(n_embd.value)]
    # do not free c_embeddings
    return embeddings


_lib.embed_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
_lib.embed_api.restype = POINTER(c_float)


def bloom_forward(ctx: c_void_p,
                  input_ids: List[int],
                  seed: c_int,
                  n_threads: c_int,
                  n_batch: c_int) -> int:
    length = len(input_ids)
    c_input_ids = (c_int * length)(*input_ids)
    token_id = _lib.forward_api(ctx, c_input_ids, length, seed, n_threads, n_batch)
    return token_id


_lib.forward_api.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
_lib.forward_api.restype = c_int

# ------------------------------------------------------------------- #
