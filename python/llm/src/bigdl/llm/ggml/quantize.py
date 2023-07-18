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
import subprocess
from bigdl.llm.utils.common import invalidInputError
import platform
from pathlib import Path


dirname, _ = os.path.split(os.path.abspath(__file__))
libs_dirname = os.path.dirname(dirname)

# ggml quantized tensor type, this is different from below file quantized type(_quantize_type)
ggml_tensor_qtype = {"sym_int4": 2,   # q4_0 in ggml
                     "asym_int4": 3,  # q4_1 in ggml
                     "sym_int5": 6,   # q5_0 in ggml
                     "asym_int5": 7,  # q5_1 in ggml
                     "sym_int8": 8}   # q8_0 in ggml

_llama_quantize_type = {"q4_0": 2,
                        "q4_1": 3,
                        "q5_0": 8,
                        "q5_1": 9,
                        "q8_0": 7}
_bloom_quantize_type = {"q4_0": 2,
                        "q4_1": 3}
_gptneox_quantize_type = {"q4_0": 2,
                          "q4_1": 3,
                          "q5_0": 8,
                          "q5_1": 9,
                          "q8_0": 7}
_starcoder_quantize_type = {"q4_0": 2,
                            "q4_1": 3,
                            "q5_0": 8,
                            "q5_1": 9,
                            "q8_0": 7}

_quantize_type = {"llama": _llama_quantize_type,
                  "bloom": _bloom_quantize_type,
                  "gptneox": _gptneox_quantize_type,
                  "starcoder": _starcoder_quantize_type}


def quantize(input_path: str, output_path: str,
             model_family: str, dtype: str='q4_0'):
    """
    Quantize ggml file to lower precision.

    :param input_path: Path of input ggml file, for example `./ggml-model-f16.bin`.
    :param output_path: Save path of output quantized model. You must pass a directory to
            save all related output. Filename of quantized model will be like
            `bigdl_llm_llama_q4_0.bin`.
    :param model_family: Which model family your input model belongs to.
            Now only `llama`/`bloom`/`gptneox`/`starcoder` are supported.
    :param dtype: Quantization method which differs in the resulting model disk size and
            inference speed. Defalut to `q4_0`. Difference model family may support
            different types, now the supported list is:
            llama : "q4_0", "q4_1", "q4_2"
            bloom : "q4_0", "q4_1"
            gptneox : "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"
            starcoder : "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"

    :return: the path str to the converted ggml binary checkpoint
    """
    invalidInputError(model_family in ['llama', 'bloom', 'gptneox', 'starcoder'],
                      "Now we only support quantization of model \
                       family('llama', 'bloom', 'gptneox', 'starcoder')",
                      "{} is not in the list.".format(model_family))
    invalidInputError(os.path.isfile(input_path),
                      "The file {} is not found".format(input_path))
    invalidInputError(os.path.isdir(output_path),
                      "The output_path {} is not a directory".format(output_path))
    # convert quantize type str into corresponding int value
    quantize_type_map = _quantize_type[model_family]
    output_filename = "bigdl_llm_{}_{}.bin".format(model_family,
                                                   dtype.lower())
    output_path = os.path.join(output_path, output_filename)
    invalidInputError(dtype.lower() in quantize_type_map, "{0} model just accept {1} now, \
                      but you pass in {2}.".format(
                      model_family,
                      list(quantize_type_map.keys()),
                      dtype))
    quantize_type = quantize_type_map[dtype]
    if platform.platform().startswith('Windows'):
        suffix = '.exe'
    else:
        suffix = ''
    quantize_args = "{0}/libs/quantize-{1}{2} {3} {4} {5}".format(libs_dirname,
                                                                  model_family,
                                                                  suffix,
                                                                  input_path,
                                                                  output_path,
                                                                  str(quantize_type))
    p = subprocess.run(quantize_args.split(), capture_output=True)
    error_message = p.stderr
    invalidInputError(not p.returncode,
                      "Fail to quantize {}, error message is {}.".format(str(input_path),
                                                                         error_message))
    return str(output_path)
