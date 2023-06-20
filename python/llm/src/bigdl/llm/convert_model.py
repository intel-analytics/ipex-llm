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
    for arg in kwargs:
        if arg not in check_args:
            return False, {arg: kwargs[arg]}
        else:
            _used_args[arg] = kwargs[arg]
    return True, _used_args


def llm_convert(model,
                outfile,
                model_family,
                outtype='int4',
                model_format="pth",
                **kwargs):
    if model_format == "pth":
        check, _used_args = _special_kwarg_check(kwargs=kwargs,
                                                 check_args=["tmp_path"])
        invalidInputError(check, f"Invaid input kwargs found: {_used_args}")
        return ggml_convert_model(input_path=model,
                                  output_path=outfile,
                                  model_family=model_family,
                                  dtype=outtype,
                                  **_used_args,
                                  )
    elif model_format == "gptq":
        invalidInputError(model.endswith(".pt"), "only support pytorch's .pt format now.")
        invalidInputError(model_family == "llama" and outtype == 'int4',
                          "Convert GPTQ models should always "
                          "specify `--model-family llama --dtype int4` in the command line.")
        check, _used_args = _special_kwarg_check(kwargs=kwargs,
                                                 check_args=["tokenizer_path"])
        invalidInputError(check, f"Invaid input kwargs found: {_used_args}")
        invalidInputError("tokenizer_path" in _used_args,
                          "The GPT-Q model requires the `tokenizer_path` parameter to be provided."
                          "Usage: convert-model --model-type gptq"
                          "--model-family llama --input-path llamaXXb-4bit.pt"
                          "--tokenizer-path tokenizer.model --output-path out.bin")
        convert_gptq2ggml(input_path=model,
                          tokenizer_path=_used_args["tokenizer_path"],
                          output_path=outfile)
        return outfile
    else:
        invalidInputError(False, f"Unsupported input model_type: {model_format}")

    return None


def main():
    parser = argparse.ArgumentParser(description='Model Convert Parameters')
    parser.add_argument('model', type=str,
                        help=("model, a path to a *directory* containing model weights"))
    parser.add_argument('-o', '--outfile', type=str, required=True,
                        help=("outfile,save path of output quantized model."))
    parser.add_argument('-x', '--model-family', type=str, required=True,
                        help=("--model-family: Which model family your input model belongs to."
                              "Now only `llama`/`bloom`/`gptneox` are supported."))
    parser.add_argument('-f', '--model-format', type=str, required=True,
                        help=("The model type to be convert to a ggml compatible file."
                              "Now only `pth`/`gptq` are supported."))
    parser.add_argument('-t', '--outtype', type=str, default="int4",
                        help="Which quantized precision will be converted.")

    # pth specific args
    parser.add_argument('-p', '--tmp-path', type=str, default=None,
                        help="Which path to store the intermediate model during the"
                        "conversion process.")

    # gptq specific args
    parser.add_argument('-k', '--tokenizer-path', type=str, default=None,
                        help="tokenizer_path, a path of tokenizer.model")
    args = parser.parse_args()
    params = vars(args)
    llm_convert(**params)
