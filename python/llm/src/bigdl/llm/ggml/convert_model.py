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


def convert_model(input_path: str,
                  output_path: str,
                  model_family: str,
                  dtype: str = 'int4',
                  tmp_path: str = '/tmp'):
    """
    Convert Hugging Face llama-like / gpt-neox-like / bloom-like model to lower precision

    :param input_path: str, path of model, for example `./llama-7b-hf`.
    :param output_path: Save path of output quantized model. Default to `None`.
            If you don't specify this parameter, quantized model will be saved in
            the same directory as the input and just replace precision with quantize_type
            like `./ggml-model-q4_0.bin`.
    :param model_family: Which model family your input model belongs to.
            Now only `llama`/`bloom`/`gptneox` are supported.
    :param dtype: Which quantized precision will be converted.
            Now only int4 supported.
    :param tmp_path: Which path to store the intermediate model during the conversion process.
    """

    dtype = dtype.lower()
    if dtype == 'int4':
        dtype = 'q4_0'

    model_name = Path(input_path).stem
    tmp_ggml_file_path = os.path.join(tmp_path, f'{model_name}_{int(time.time())}')
    _convert_to_ggml(model_path=input_path,
                     outfile_dir=tmp_ggml_file_path,
                     model_family=model_family,
                     outtype="fp16")

    tmp_ggml_file_path = next(Path(tmp_ggml_file_path).iterdir())

    return quantize(input_path=tmp_ggml_file_path,
                    output_path=output_path,
                    model_family=model_family,
                    dtype=dtype)
