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
from ipex_llm.utils.common import invalidInputError
import platform
from pathlib import Path


dirname, _ = os.path.split(os.path.abspath(__file__))
libs_dirname = os.path.dirname(dirname)

# ggml quantized tensor type, this is different from below file quantized type(_quantize_type)
ggml_tensor_qtype = {"sym_int4": 2,   # q4_0 in ggml
                     "asym_int4": 3,  # q4_1 in ggml
                     "sym_int5": 6,   # q5_0 in ggml
                     "asym_int5": 7,  # q5_1 in ggml
                     "sym_int8": 8,   # q8_0 in ggml
                     "nf4": 10,
                     "nf3": 11,
                     "fp16": 12,
                     "fp8_e4m3": 15,      # fp8 in e4m3 format
                     "fp4": 16,
                     "mixed_fp4": 17,     # Mixture of Formats Quantization 4 bits
                     "mixed_fp8": 18,     # Mixture of Formats Quantization 8 bits
                     "fp8_e5m2": 19,      # fp8 in e5m2 format
                     "fp8": 19,           # fp8 in e5m2 format
                     "bf16": 20,
                     "gguf_iq2_xxs": 21,
                     "gguf_iq2_xs": 22,
                     "q2_k": 23,
                     "gguf_iq1_s": 24,
                     "gguf_iq1_m": 25,
                     "q6_k": 26,
                     "q4_k": 27,
                     "q5_k": 28,
                     "fp6": 29,
                     "fp6_k": 30,
                     "sym_int4_rtn": 31,
                     "sym_int8_rtn": 32,
                     "asym_int4_rtn": 33,
                     "woq_int4": 34,
                     }

# mixed precison from llama.cpp
gguf_mixed_qtype = {"gguf_q4k_s": 101,
                    "gguf_q4k_m": 102}

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
            llama : "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"
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
