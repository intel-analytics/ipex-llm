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

# MIT License

# Copyright (c) 2023 Georgi Gerganov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.utils.convert_util import *
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
        vocab = load_vocab(vocab_dir)
    model = model_plus.model
    model = do_necessary_conversions(model)
    output_type = pick_output_type(model, outtype)
    model = convert_to_output_type(model, output_type)
    params = Params.guessed(model, output_type)
    outfile_path = default_outfile(outfile_dir, params)
    OutputFile.write_all(outfile_path, params, model, vocab)


def _convert_gptneox(model_path, outfile_dir, outtype):
    _convert_gptneox_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_bloomz(model_path, outfile_dir, outtype):
    _convert_bloomz_hf_to_ggml(model_path, outfile_dir, outtype)


def _convert_to_ggml(model_path: str, outfile_dir: str, model_family: str = 'llama', outtype: str="f16"):
    """
    Convert Hugging Face llama-like / gpt-neox-like / bloom-like model to ggml format.

    :param model_path: str, path of model, for example `./llama-7b-hf`.
    :param outfile_dir: str, the directory to save ggml compatible file, for example `./models`.
    :param model_family: Which model family your input model belongs to. Default to `llama`.
            Now only `llama`/`bloomz`/`gptneox` are supported.
    :param outtype: specify the output format. Defalut to `f16`. Now `f32`/`f16` are supported.
    """
    invalidInputError(model_family in ['llama', 'bloomz', 'gptneox'],
                      "Now we only support quantization of model \
                       family('llama', 'bloomz', 'gptneox')",
                      "{} is not in the list.".format(model_family))
    invalidInputError(os.path.exists(model_path),
                      "The file {} was not found".format(model_path))

    # make sure the output directory exists
    os.makedirs(outfile_dir, exist_ok=True)

    if model_family == 'llama':
        _convert_llama(model_path, outfile_dir, outtype)
    if model_family == 'gptneox':
        _convert_gptneox(model_path, outfile_dir, outtype)
    if model_family == 'bloomz':
        _convert_bloomz(model_path, outfile_dir, outtype)
