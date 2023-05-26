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


dirname, _ = os.path.split(os.path.abspath(__file__))
bin_dirname = os.path.dirname(dirname)

_llama_quantize_type = {"q4_0": 2,
                        "q4_1": 3,
                        "q4_2": 5}
_bloomz_quantize_type = {"q4_0": 2,
                         "q4_1": 3}
_gptneox_quantize_type = {"q4_0": 2,
                          "q4_1": 3,
                          "q4_2": 5,
                          "q5_0": 8,
                          "q5_1": 9,
                          "q8_0": 7}

_quantize_type = {"llama": _llama_quantize_type,
                  "bloomz": _bloomz_quantize_type,
                  "gptneox": _gptneox_quantize_type}

_valid_types = set(list(_llama_quantize_type.keys()) + list(_bloomz_quantize_type.keys()) + list(_gptneox_quantize_type.keys()))


def quantize(model_path: str, quantize_path: str=None, quantize_type: str='q4_0', model_family: str = 'llama'):
    """
    Quantize ggml file to lower precision.

    :param model_path: Path of input ggml file, for example `./ggml-model-f16.bin`.
    :param quantize_path: Save path of output quantized model. Default to `None`. 
            If you don't specify this parameter, quantized model will be saved in
            the same directory as the input and just replace precision with quantize_type
            like `./ggml-model-q4_0.bin`.
    :param quantize_type: Quantization method which differs in the resulting model disk size and
            inference speed. Defalut to `q4_0`. Difference model family may support different types,
            now the supported list is:
            llama : "q4_0", "q4_1", "q4_2"
            bloomz : "q4_0", "q4_1"
            gptneox : "q4_0", "q4_1", "q4_2", "q5_0", "q5_1", "q8_0"
    :param model_family: Which model family your input model belongs to. Default to `llama`.
            Now only `llama`/`bloomz`/`gptneox` are supported.
    """
    invalidInputError(model_family in ['llama', 'bloomz', 'gptneox'],
                      "Now we only support quantization of model family('llama', 'bloomz', 'gptneox')",
                      "{} is not in the list.".format(model_family))
    invalidInputError(os.path.isfile(model_path),
                      "The file {} was not found".format(model_path))
    # TODO : multi input model path
    if quantize_path is None:
        quantize_path = model_path.replace("f16", quantize_type)
    # convert quantize type str into corresponding int value
    quantize_type_map = _quantize_type[model_family]
    invalidInputError(quantize_type in quantize_type_map,
                      "{0} model just accept {1} now, but you pass in {2}.".format(
                        model_family,
                        list(quantize_type_map.keys()),
                        quantize_type))
    quantize_type = quantize_type_map[quantize_type]
    quantize_args = "{0}/bin/quantize-{1} {2} {3} {4}".format(bin_dirname,
                                                              model_family,
                                                              model_path,
                                                              quantize_path,
                                                              str(quantize_type))
    p = subprocess.Popen(quantize_args.split())
    p.communicate()
    invalidInputError(not p.returncode,
                      "Fail to quantize {}.".format(str(model_path)))
