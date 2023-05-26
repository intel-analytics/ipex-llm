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


def quantize(input_filename: str, output_filename: str=None, ftype: str='q4_0', model_family: str = 'llama'):
    invalidInputError(model_family in ['llama', 'bloomz', 'gptneox'],
                      "Now we only support quantization of model family('llama', 'bloomz', 'gptneox')",
                      "{} is not in the list.".format(model_family))
    # TODO : auto construction of output filename

    # TODO: convert str type into number based on each model
    if ftype == 'q4_0':
        ftype = 2

    quantize_args = "{0}/bin/quantize-{1} {2} {3} {4}".format(dirname, model_family, input_filename, output_filename, str(ftype))
    p = subprocess.Popen(quantize_args.split())
    p.communicate()
    invalidInputError(not p.returncode,
                      "Fail to quantize {}.".format(str(input_filename)))
