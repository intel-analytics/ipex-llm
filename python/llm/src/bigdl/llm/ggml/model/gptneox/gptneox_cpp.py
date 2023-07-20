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
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    pointer,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils import get_shared_lib_info


# Load the library
def _load_shared_library(lib_base_name: str):
    _base_path, _lib_paths = get_shared_lib_info(lib_base_name=lib_base_name)

    if "GPTNEOX_CPP_LIB" in os.environ:
        lib_base_name = os.environ["GPTNEOX_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                invalidInputError(False, f"Failed to load shared library '{_lib_path}': {e}.")

    invalidInputError(False, f"Shared library with base name '{lib_base_name}' not found.")


# Specify the base name of the shared library to load
_lib_base_name = "gptneox"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# C types
GPTNEOX_FILE_VERSION = c_int(1)
GPTNEOX_FILE_MAGIC = b"ggjt"
GPTNEOX_FILE_MAGIC_UNVERSIONED = b"ggml"
# GPTNEOX_SESSION_MAGIC = b"ggsn"
# GPTNEOX_SESSION_VERSION = c_int(1)

gptneox_context_p = c_void_p


gptneox_token = c_int
gptneox_token_p = POINTER(gptneox_token)


class gptneox_token_data(Structure):
    _fields_ = [
        ("id", gptneox_token),  # token id
        ("logit", c_float),  # log-odds of the token
        ("p", c_float),  # probability of the token
    ]


gptneox_token_data_p = POINTER(gptneox_token_data)


class gptneox_token_data_array(Structure):
    _fields_ = [
        ("data", gptneox_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


gptneox_token_data_array_p = POINTER(gptneox_token_data_array)

gptneox_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)


class gptneox_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),  # text context
        ("n_parts", c_int),  # -1 for default
        # ("n_gpu_layers", c_int),  # number of layers to store in VRAM
        ("seed", c_int),  # RNG seed, 0 for random
        ("f16_kv", c_bool),  # use fp16 for KV cache
        (
            "logits_all",
            c_bool,
        ),  # the gptneox_eval() call computes all logits, not just the last one
        ("vocab_only", c_bool),  # only load the vocabulary, no weights
        ("use_mmap", c_bool),  # use mmap if possible
        ("use_mlock", c_bool),  # force system to keep model in RAM
        ("embedding", c_bool),  # embedding mode only
        # called with a progress value between 0 and 1, pass NULL to disable
        ("progress_callback", gptneox_progress_callback),
        # context pointer passed to the progress callback
        ("progress_callback_user_data", c_void_p),
    ]


gptneox_context_params_p = POINTER(gptneox_context_params)

GPTNEOX_FTYPE_ALL_F32 = c_int(0)
GPTNEOX_FTYPE_MOSTLY_F16 = c_int(1)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q4_0 = c_int(2)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q4_1 = c_int(3)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(
    4
)  # tok_embeddings.weight and output.weight are F16
GPTNEOX_FTYPE_MOSTLY_Q4_2 = c_int(5)  # except 1d tensors
# GPTNEOX_FTYPE_MOSTYL_Q4_3 = c_int(6)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q8_0 = c_int(7)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q5_0 = c_int(8)  # except 1d tensors
GPTNEOX_FTYPE_MOSTLY_Q5_1 = c_int(9)  # except 1d tensors

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

# Functions


def gptneox_context_default_params() -> gptneox_context_params:
    return _lib.gptneox_context_default_params()


_lib.gptneox_context_default_params.argtypes = []
_lib.gptneox_context_default_params.restype = gptneox_context_params


def gptneox_mmap_supported() -> bool:
    return _lib.gptneox_mmap_supported()


_lib.gptneox_mmap_supported.argtypes = []
_lib.gptneox_mmap_supported.restype = c_bool


def gptneox_mlock_supported() -> bool:
    return _lib.gptneox_mlock_supported()


_lib.gptneox_mlock_supported.argtypes = []
_lib.gptneox_mlock_supported.restype = c_bool


# Various functions for loading a ggml gptneox model.
# Allocate (almost) all memory needed for the model.
# Return NULL on failure
def gptneox_init_from_file(
    path_model: bytes, params: gptneox_context_params
) -> gptneox_context_p:
    return _lib.gptneox_init_from_file(path_model, pointer(params))


_lib.gptneox_init_from_file.argtypes = [c_char_p, gptneox_context_params_p]
_lib.gptneox_init_from_file.restype = gptneox_context_p


# Frees all allocated memory
def gptneox_free(ctx: gptneox_context_p):
    _lib.gptneox_free(ctx)


_lib.gptneox_free.argtypes = [gptneox_context_p]
_lib.gptneox_free.restype = None


# TODO: not great API - very likely to change
# Returns 0 on success
# nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(),
# else the number given
def gptneox_model_quantize(
    fname_inp: bytes, fname_out: bytes, ftype: c_int, nthread: c_int
) -> c_int:
    return _lib.gptneox_model_quantize(fname_inp, fname_out, ftype, nthread)


_lib.gptneox_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
_lib.gptneox_model_quantize.restype = c_int


def gptneox_model_copy(
    fname_inp: bytes, fname_out: bytes, ftype: c_int
) -> c_int:
    return _lib.gptneox_model_copy(fname_inp, fname_out, ftype)

_lib.gptneox_model_copy.argtypes = [c_char_p, c_char_p, c_int]
_lib.gptneox_model_copy.restype = c_int


# Apply a LoRA adapter to a loaded model
# path_base_model is the path to a higher quality model to use as a base for
# the layers modified by the adapter. Can be NULL to use the current loaded model.
# The model needs to be reloaded before applying a new adapter, otherwise the adapter
# will be applied on top of the previous one
# Returns 0 on success
def gptneox_apply_lora_from_file(
    ctx: gptneox_context_p,
    path_lora: c_char_p,
    path_base_model: c_char_p,
    n_threads: c_int,
) -> c_int:
    return _lib.gptneox_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


_lib.gptneox_apply_lora_from_file.argtypes = [gptneox_context_p, c_char_p, c_char_p, c_int]
_lib.gptneox_apply_lora_from_file.restype = c_int


# Returns the number of tokens in the KV cache
def gptneox_get_kv_cache_token_count(ctx: gptneox_context_p) -> c_int:
    return _lib.gptneox_get_kv_cache_token_count(ctx)


_lib.gptneox_get_kv_cache_token_count.argtypes = [gptneox_context_p]
_lib.gptneox_get_kv_cache_token_count.restype = c_int


# Sets the current rng seed.
def gptneox_set_rng_seed(ctx: gptneox_context_p, seed: c_int):
    return _lib.gptneox_set_rng_seed(ctx, seed)


_lib.gptneox_set_rng_seed.argtypes = [gptneox_context_p, c_int]
_lib.gptneox_set_rng_seed.restype = None


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
def gptneox_get_state_size(ctx: gptneox_context_p) -> c_size_t:
    return _lib.gptneox_get_state_size(ctx)


_lib.gptneox_get_state_size.argtypes = [gptneox_context_p]
_lib.gptneox_get_state_size.restype = c_size_t


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
def gptneox_copy_state_data(
    ctx: gptneox_context_p, dst  # type: Array[c_uint8]
) -> int:
    return _lib.gptneox_copy_state_data(ctx, dst)


_lib.gptneox_copy_state_data.argtypes = [gptneox_context_p, c_uint8_p]
_lib.gptneox_copy_state_data.restype = c_size_t


# Set the state reading from the specified address
# Returns the number of bytes read
def gptneox_set_state_data(
    ctx: gptneox_context_p, src  # type: Array[c_uint8]
) -> int:
    return _lib.gptneox_set_state_data(ctx, src)


_lib.gptneox_set_state_data.argtypes = [gptneox_context_p, c_uint8_p]
_lib.gptneox_set_state_data.restype = c_size_t


# Save/load session file
def gptneox_load_session_file(
    ctx: gptneox_context_p,
    path_session: bytes,
    tokens_out,  # type: Array[gptneox_token]
    n_token_capacity: c_size_t,
    n_token_count_out,  # type: _Pointer[c_size_t]
) -> c_size_t:
    return _lib.gptneox_load_session_file(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out
    )


_lib.gptneox_load_session_file.argtypes = [
    gptneox_context_p,
    c_char_p,
    gptneox_token_p,
    c_size_t,
    c_size_t_p,
]
_lib.gptneox_load_session_file.restype = c_size_t


def gptneox_save_session_file(
    ctx: gptneox_context_p,
    path_session: bytes,
    tokens,  # type: Array[gptneox_token]
    n_token_count: c_size_t,
) -> c_size_t:
    return _lib.gptneox_save_session_file(ctx, path_session, tokens, n_token_count)


_lib.gptneox_save_session_file.argtypes = [
    gptneox_context_p,
    c_char_p,
    gptneox_token_p,
    c_size_t,
]
_lib.gptneox_save_session_file.restype = c_size_t


# Run the gptneox inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
def gptneox_eval(
    ctx: gptneox_context_p,
    tokens,  # type: Array[gptneox_token]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> c_int:
    return _lib.gptneox_eval(ctx, tokens, n_tokens, n_past, n_threads)


_lib.gptneox_eval.argtypes = [gptneox_context_p, gptneox_token_p, c_int, c_int, c_int]
_lib.gptneox_eval.restype = c_int


# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
def gptneox_tokenize(
    ctx: gptneox_context_p,
    text: bytes,
    tokens,  # type: Array[gptneox_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> int:
    return _lib.gptneox_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


_lib.gptneox_tokenize.argtypes = [gptneox_context_p, c_char_p, gptneox_token_p, c_int, c_bool]
_lib.gptneox_tokenize.restype = c_int


def gptneox_n_vocab(ctx: gptneox_context_p) -> c_int:
    return _lib.gptneox_n_vocab(ctx)


_lib.gptneox_n_vocab.argtypes = [gptneox_context_p]
_lib.gptneox_n_vocab.restype = c_int


def gptneox_n_ctx(ctx: gptneox_context_p) -> c_int:
    return _lib.gptneox_n_ctx(ctx)


_lib.gptneox_n_ctx.argtypes = [gptneox_context_p]
_lib.gptneox_n_ctx.restype = c_int


def gptneox_n_embd(ctx: gptneox_context_p) -> c_int:
    return _lib.gptneox_n_embd(ctx)


_lib.gptneox_n_embd.argtypes = [gptneox_context_p]
_lib.gptneox_n_embd.restype = c_int


# Token logits obtained from the last call to gptneox_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
def gptneox_get_logits(
    ctx: gptneox_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.gptneox_get_logits(ctx)


_lib.gptneox_get_logits.argtypes = [gptneox_context_p]
_lib.gptneox_get_logits.restype = c_float_p


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
def gptneox_get_embeddings(
    ctx: gptneox_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.gptneox_get_embeddings(ctx)


_lib.gptneox_get_embeddings.argtypes = [gptneox_context_p]
_lib.gptneox_get_embeddings.restype = c_float_p


# Token Id -> String. Uses the vocabulary in the provided context
def gptneox_token_to_str(ctx: gptneox_context_p, token: gptneox_token) -> bytes:
    return _lib.gptneox_token_to_str(ctx, token)


_lib.gptneox_token_to_str.argtypes = [gptneox_context_p, gptneox_token]
_lib.gptneox_token_to_str.restype = c_char_p


# String -> Token Id. Uses the vocabulary in the provided context
def gptneox_str_to_token(ctx: gptneox_context_p, input_str: c_char_p):
    return _lib.gptneox_str_to_token(ctx, input_str)


_lib.gptneox_str_to_token.argtypes = [gptneox_context_p, c_char_p]
_lib.gptneox_str_to_token.restype = gptneox_token

# TODO: improve the last_n_tokens interface ?
# def gptneox_sample_top_p_top_k(ctx: gptneox_context_p, last_n_tokens_data: gptneox_token,
#                                last_n_tokens_size: c_int, top_k: c_int, top_p: c_float,
#                                temp: c_float, repeat_penalty: c_float):
#     return _lib.gptneox_sample_top_p_top_k(ctx, last_n_tokens_data, last_n_tokens_size,
#                                            top_k, top_p, temp, repeat_penalty)


# _lib.gptneox_sample_top_p_top_k.argtypes = [gptneox_context_p, gptneox_token,
# c_int, c_int, c_float, c_float, c_float]
# _lib.gptneox_sample_top_p_top_k.restype = gptneox_token

# Special tokens


def gptneox_token_bos() -> gptneox_token:
    return _lib.gptneox_token_bos()


_lib.gptneox_token_bos.argtypes = []
_lib.gptneox_token_bos.restype = gptneox_token


def gptneox_token_eos() -> gptneox_token:
    return _lib.gptneox_token_eos()


_lib.gptneox_token_eos.argtypes = []
_lib.gptneox_token_eos.restype = gptneox_token


# def gptneox_token_nl() -> gptneox_token:
#     return _lib.gptneox_token_nl()


# _lib.gptneox_token_nl.argtypes = []
# _lib.gptneox_token_nl.restype = gptneox_token


# Sampling functions


def gptneox_get_candidates(
    ctx: gptneox_context_p,
    n_vocab: c_int,
    logits: c_float_p,
):
    return _lib.gptneox_get_candidates(
        ctx, n_vocab, logits
    )


_lib.gptneox_get_candidates.argtypes = [
    gptneox_context_p,
    c_int,
    c_float_p
]
_lib.gptneox_get_candidates.restype = gptneox_token_data_array


# @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858,
# with negative logit fix.
def gptneox_sample_repetition_penalty(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    last_tokens_data,  # type: Array[gptneox_token]
    last_tokens_size: c_int,
    penalty: c_float,
):
    return _lib.gptneox_sample_repetition_penalty(
        ctx, candidates, last_tokens_data, last_tokens_size, penalty
    )


_lib.gptneox_sample_repetition_penalty.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    gptneox_token_p,
    c_int,
    c_float,
]
_lib.gptneox_sample_repetition_penalty.restype = None


# @details Frequency and presence penalties described in OpenAI API
# https://platform.openai.com/docs/api-reference/parameter-details.
def gptneox_sample_frequency_and_presence_penalties(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    last_tokens_data,  # type: Array[gptneox_token]
    last_tokens_size: c_int,
    alpha_frequency: c_float,
    alpha_presence: c_float,
):
    return _lib.gptneox_sample_frequency_and_presence_penalties(
        ctx,
        candidates,
        last_tokens_data,
        last_tokens_size,
        alpha_frequency,
        alpha_presence,
    )


_lib.gptneox_sample_frequency_and_presence_penalties.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    gptneox_token_p,
    c_int,
    c_float,
    c_float,
]
_lib.gptneox_sample_frequency_and_presence_penalties.restype = None


# @details Sorts candidate tokens by their logits in descending order and
# calculate probabilities based on logits.
def gptneox_sample_softmax(
    ctx: gptneox_context_p, candidates  # type: _Pointer[gptneox_token_data]
):
    return _lib.gptneox_sample_softmax(ctx, candidates)


_lib.gptneox_sample_softmax.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
]
_lib.gptneox_sample_softmax.restype = None


# @details Top-K sampling described in academic paper
# "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
def gptneox_sample_top_k(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    k: c_int,
    min_keep: c_size_t,
):
    return _lib.gptneox_sample_top_k(ctx, candidates, k, min_keep)


_lib.gptneox_sample_top_k.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_int,
    c_size_t,
]
_lib.gptneox_sample_top_k.restype = None


# @details Nucleus sampling described in academic paper
# "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
def gptneox_sample_top_p(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.gptneox_sample_top_p(ctx, candidates, p, min_keep)


_lib.gptneox_sample_top_p.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.gptneox_sample_top_p.restype = None


# @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
def gptneox_sample_tail_free(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    z: c_float,
    min_keep: c_size_t,
):
    return _lib.gptneox_sample_tail_free(ctx, candidates, z, min_keep)


_lib.gptneox_sample_tail_free.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.gptneox_sample_tail_free.restype = None


# @details Locally Typical Sampling implementation described in the paper
# https://arxiv.org/abs/2202.00666.
def gptneox_sample_typical(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.gptneox_sample_typical(ctx, candidates, p, min_keep)


_lib.gptneox_sample_typical.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.gptneox_sample_typical.restype = None


def gptneox_sample_temperature(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    temp: c_float,
):
    return _lib.gptneox_sample_temperature(ctx, candidates, temp)


_lib.gptneox_sample_temperature.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
]
_lib.gptneox_sample_temperature.restype = None


# @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966.
# Uses tokens instead of words.
# @param candidates A vector of `gptneox_token_data` containing the candidate tokens,
# their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated
# text. A higher value corresponds to more surprising or less predictable text, while a lower value
# corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and
# observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
# updated more quickly, while a smaller learning rate will result in slower updates.
# @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value
# that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`.
# In the paper, they use `m = 100`, but you can experiment with different values to see
# how it affects the performance of the algorithm.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy
# (`2 * tau`) and is updated in the algorithm based on the error between the target and
# observed surprisal.
def gptneox_sample_token_mirostat(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    tau: c_float,
    eta: c_float,
    m: c_int,
    mu,  # type: _Pointer[c_float]
) -> gptneox_token:
    return _lib.gptneox_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)


_lib.gptneox_sample_token_mirostat.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
    c_float,
    c_int,
    c_float_p,
]
_lib.gptneox_sample_token_mirostat.restype = gptneox_token


# @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966.
# Uses tokens instead of words.
# @param candidates A vector of `gptneox_token_data` containing the candidate tokens,
# their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated
# text. A higher value corresponds to more surprising or less predictable text, while a lower value
# corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and
# observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
# updated more quickly, while a smaller learning rate will result in slower updates.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy
# (`2 * tau`) and is updated in the algorithm based on the error between the target and
# observed surprisal.
def gptneox_sample_token_mirostat_v2(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
    tau: c_float,
    eta: c_float,
    mu,  # type: _Pointer[c_float]
) -> gptneox_token:
    return _lib.gptneox_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)


_lib.gptneox_sample_token_mirostat_v2.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
    c_float,
    c_float,
    c_float_p,
]
_lib.gptneox_sample_token_mirostat_v2.restype = gptneox_token


# @details Selects the token with the highest probability.
def gptneox_sample_token_greedy(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
) -> gptneox_token:
    return _lib.gptneox_sample_token_greedy(ctx, candidates)


_lib.gptneox_sample_token_greedy.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
]
_lib.gptneox_sample_token_greedy.restype = gptneox_token


# @details Randomly selects a token from the candidates based on their probabilities.
def gptneox_sample_token(
    ctx: gptneox_context_p,
    candidates,  # type: _Pointer[gptneox_token_data_array]
) -> gptneox_token:
    return _lib.gptneox_sample_token(ctx, candidates)


_lib.gptneox_sample_token.argtypes = [
    gptneox_context_p,
    gptneox_token_data_array_p,
]
_lib.gptneox_sample_token.restype = gptneox_token


# Performance information


def gptneox_print_timings(ctx: gptneox_context_p):
    _lib.gptneox_print_timings(ctx)


_lib.gptneox_print_timings.argtypes = [gptneox_context_p]
_lib.gptneox_print_timings.restype = None


def gptneox_reset_timings(ctx: gptneox_context_p):
    _lib.gptneox_reset_timings(ctx)


_lib.gptneox_reset_timings.argtypes = [gptneox_context_p]
_lib.gptneox_reset_timings.restype = None


# Print system information
def gptneox_print_system_info() -> bytes:
    return _lib.gptneox_print_system_info()


_lib.gptneox_print_system_info.argtypes = []
_lib.gptneox_print_system_info.restype = c_char_p
