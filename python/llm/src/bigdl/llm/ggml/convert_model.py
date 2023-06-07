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
import os
import time
from pathlib import Path
from bigdl.llm.ggml.convert import _convert_to_ggml
from bigdl.llm.ggml.quantize import quantize
from bigdl.llm.utils.common import invalidInputError
import argparse
import tempfile


def convert_model(input_path: str,
                  output_dir: str,
                  model_family: str,
                  dtype: str = 'int4',
                  tmp_path: str = None):
    """
    Convert Hugging Face llama-like / gpt-neox-like / bloom-like model to lower precision

    :param input_path: A path to a *directory* containing model weights saved using
            `PreTrainedModel.save_pretrained` in transformers, for example `./llama-7b-hf`.
    :param output_dir: Save path of output quantized model. You must pass a *directory* to
            save all related output.
    :param model_family: Which model family your input model belongs to.
            Now only `llama`/`bloom`/`gptneox` are supported.
    :param dtype: Which quantized precision will be converted.
            Now only int4 is supported.
    :param tmp_path: Which path to store the intermediate model during the conversion process.
            Default to `None` so that intermediate model will not be saved.

    :return: the path string to the converted lower precision checkpoint.
    """

    dtype = dtype.lower()
    # make sure directory exists
    os.makedirs(output_dir, exist_ok=True)
    # check input value
    invalidInputError(model_family in ['llama', 'bloom', 'gptneox'],
                      "Now we only support quantization of model \
                       family('llama', 'bloom', 'gptneox')",
                      "{} is not in the list.".format(model_family))
    invalidInputError(os.path.isdir(output_dir),
                      "The output_dir {} was not a directory".format(output_dir))
    invalidInputError(dtype == 'int4',
                      "Now only int4 is supported.")
    # check for input_path
    invalidInputError(os.path.exists(input_path),
                      "The input path {} was not found".format(input_path))
    invalidInputError(os.path.isdir(input_path),
                      "The input path {} was not a directory".format(input_path))
    # shall we support model_id or just model directory?

    if dtype == 'int4':
        dtype = 'q4_0'

    if tmp_path is not None:
        model_name = Path(input_path).stem
        tmp_ggml_file_path = os.path.join(tmp_path, f'{model_name}_{int(time.time())}')
        _convert_to_ggml(model_path=input_path,
                         outfile_dir=tmp_ggml_file_path,
                         model_family=model_family,
                         outtype="fp16")
        tmp_ggml_file_path = next(Path(tmp_ggml_file_path).iterdir())
        return quantize(input_path=tmp_ggml_file_path,
                        output_dir=output_dir,
                        model_family=model_family,
                        dtype=dtype)
    else:
        with tempfile.TemporaryDirectory() as tmp_ggml_file_path:
            _convert_to_ggml(model_path=input_path,
                             outfile_dir=tmp_ggml_file_path,
                             model_family=model_family,
                             outtype="fp16")
            tmp_ggml_file_path = next(Path(tmp_ggml_file_path).iterdir())
            return quantize(input_path=tmp_ggml_file_path,
                            output_dir=output_dir,
                            model_family=model_family,
                            dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description='Model Convert Parameters')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help=("input_path, a path to a *directory* containing model weights"))
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help=("output_dir,save path of output quantized model."))
    parser.add_argument('-f', '--model_family', type=str, required=True,
                        help=("model_family: Which model family your input model belongs to."
                              "Now only `llama`/`bloom`/`gptneox` are supported."))
    parser.add_argument('-t', '--dtype', type=str, default="int4",
                        help="Which quantized precision will be converted.")
    parser.add_argument('-p', '--tmp_path', type=str, default=None,
                        help="Which path to store the intermediate model during the"
                        "conversion process.")
    args = parser.parse_args()
    params = vars(args)
    convert_model(**params)
