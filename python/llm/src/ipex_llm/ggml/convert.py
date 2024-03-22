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

from ipex_llm.utils.common import invalidInputError
from ipex_llm.utils.convert_util import *
from pathlib import Path
import os


def _convert_llama(model_path, outfile_dir, outtype):
    model_path = Path(model_path)
    outfile_dir = Path(outfile_dir)
    model_plus = load_some_model(model_path)
    if model_plus.vocab is not None:
        vocab = model_plus.vocab
    else:
        vocab_dir = model_plus.paths[0].parent
        vocab = load_vocab(vocab_dir, vocabtype='spm')
    params = Params.load(model_plus)
    model = model_plus.model
    model = do_necessary_conversions(model, params)
    output_type = pick_output_type(model, outtype)
    model = convert_to_output_type(model, output_type)
    outfile_path = default_outfile([outfile_dir], output_type)
    OutputFile.write_all(outfile_path, params, output_type, model, vocab)


def _convert_gptneox(model_path, outfile_dir, outtype):
    _convert_gptneox_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_bloom(model_path, outfile_dir, outtype):
    _convert_bloom_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_starcoder(model_path, outfile_dir, outtype):
    _convert_starcoder_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_chatglm(model_path, outfile_dir, outtype):
    return _convert_chatglm_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_to_ggml(model_path: str, outfile_dir: str,
                     model_family: str = 'llama', outtype: str="fp16"):
    """
    Convert Hugging Face llama-like / gpt-neox-like / bloom-like model to ggml format.

    :param input_path: Path to a *directory*  for huggingface checkpoint that are directly
            pulled from huggingface hub, for example `./llama-7b-hf`. This should be a dir
            path that contains: weight bin, tokenizer config, tokenizer.model (required for
            llama) and added_tokens.json (if applied).
            For lora finetuned model, the path should be pointed to a merged weight.
    :param outfile_dir: str, the directory to save ggml compatible file, for example `./models`.
    :param model_family: Which model family your input model belongs to. Default to `llama`.
            Now only `llama`/`bloom`/`gptneox`/`starcoder` are supported.
    :param outtype: specify the output format. Defalut to `fp16`. Now `fp32`/`fp16` are supported.
    """
    invalidInputError(model_family in ['llama', 'bloom', 'gptneox', 'starcoder'],
                      "Now we only support quantization of model \
                       family('llama', 'bloom', 'gptneox', 'starcoder')",
                      "{} is not in the list.".format(model_family))
    invalidInputError(os.path.exists(model_path),
                      "The file {} was not found".format(model_path))
    invalidInputError(outtype in ['fp32', 'fp16'],
                      "Now we only support converting to 'fp32'/'fp16' format",
                      "{} is not in the list.".format(outtype))

    # make sure the output directory exists
    os.makedirs(outfile_dir, exist_ok=True)

    outtype = outtype.replace('p', '')
    print("It may takes several minutes to load the original model, please wait...")
    if model_family == 'llama':
        _convert_llama(model_path, outfile_dir, outtype)
    if model_family == 'gptneox':
        _convert_gptneox(model_path, outfile_dir, outtype)
    if model_family == 'bloom':
        _convert_bloom(model_path, outfile_dir, outtype)
    if model_family == 'starcoder':
        _convert_starcoder(model_path, outfile_dir, outtype)
