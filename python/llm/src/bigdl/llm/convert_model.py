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


from bigdl.llm.ggml.convert_model import convert_model as ggml_convert_model
from bigdl.llm.gptq.convert.convert_gptq_to_ggml import convert_gptq2ggml
from bigdl.llm.utils.common import invalidInputError
import argparse


def _special_kwarg_check(kwargs, check_args):
    _used_args = {}
    for arg in check_args:
        if arg not in kwargs:
            return False, {arg, kwargs[arg]}
        else:
            _used_args[arg] = kwargs[arg]
    return True, _used_args


def convert_model(input_path,
                  output_path,
                  model_family,
                  dtype='int4',
                  model_type="pth",
                  **kwargs):
    if model_type == "pth":
        check, _used_args = _special_kwarg_check(kwargs=kwargs,
                                                 check_args=["tmp_path"])
        invalidInputError(check, f"Invaid input kwargs found: {_used_args}")
        ggml_convert_model(input_path=input_path,
                           output_path=output_path,
                           model_family=model_family,
                           dtype=dtype,
                           **_used_args,
                           )
    elif model_type == "gptq":
        invalidInputError(input_path.endswith(".pt"), "only support pytorch's .pt format now.")
        invalidInputError(model_family == "llama" and dtype == 'int4',
                          "Convert GPTQ models should always "
                          "specify `--model_family llama --dtype int4` in the command line.")
        check, _used_args = _special_kwarg_check(kwargs=kwargs,
                                                 check_args=["tokenizer_path"])
        invalidInputError(check, f"Invaid input kwargs found: {_used_args}")
        invalidInputError("tokenizer_path" in _used_args,
                          "The GPT-Q model requires the `tokenizer_path` parameter to be provided."
                          "Usage: convert-model --model_type gptq"
                          "--model_family llama --input_path llamaXXb-4bit.pt"
                          "--tokenizer_path tokenizer.model --output_path out.bin")
        convert_gptq2ggml(input_path=input_path,
                          tokenizer_path=_used_args["tokenizer_path"],
                          output_path=output_path)
    else:
        invalidInputError(False, f"Unsupported input model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Model Convert Parameters')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help=("input_path, a path to a *directory* containing model weights"))
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help=("output_path,save path of output quantized model."))
    parser.add_argument('-x', '--model_family', type=str, required=True,
                        help=("model_family: Which model family your input model belongs to."
                              "Now only `llama`/`bloom`/`gptneox` are supported."))
    parser.add_argument('-m', '--model_type', type=str, required=True,
                        help=("The model type to be convert to a ggml compatible file."
                              "Now only `pth`/`gptq` are supported."))
    parser.add_argument('-t', '--dtype', type=str, default="int4",
                        help="Which quantized precision will be converted.")

    # pth specific args
    parser.add_argument('-p', '--tmp_path', type=str, default=None,
                        help="Which path to store the intermediate model during the"
                        "conversion process.")

    # gptq specific args
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help="tokenizer_path, a path of tokenizer.model")
    args = parser.parse_args()
    params = vars(args)
    convert_model(**params)
