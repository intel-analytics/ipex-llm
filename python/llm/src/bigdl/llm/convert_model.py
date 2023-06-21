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
import tempfile
import os
import time
from pathlib import Path


def _special_kwarg_check(kwargs, check_args):
    _used_args = {}
    for arg in kwargs:
        if arg in check_args:
            _used_args[arg] = kwargs[arg]
    return True, _used_args


def llm_convert(model,
                outfile,
                model_family,
                outtype='int4',
                model_format="pth",
                lora_id_or_path=None,
                **kwargs):
    """
    This function is able to:

        1. Convert Hugging Face llama-like / gpt-neox-like / bloom-like / starcoder-like
           PyTorch model to lower precision in BigDL-LLM optimized GGML format.
        2. Convert Hugging Face GPTQ format llama-like model to BigDL-LLM optimized
           GGML format.

    :param model: Path to a **directory**:

           1. If ``model_format='pth'``, the folder should be a Hugging Face checkpoint
              that is directly pulled from Hugging Face hub, for example ``./llama-7b-hf``.
              This should be a dir path that contains: weight bin, tokenizer config,
              tokenizer.model (required for llama) and added_tokens.json (if applied).
              For lora finetuned model, the path should be pointed to a merged weight.
           2. If ``model_format='gptq'``, the folder should be be a Hugging Face checkpoint
              in GPTQ format, which contains weights in pytorch's .pt format,
              and ``tokenizer.model``.

    :param outfile: Save path of output quantized model. You must pass a **directory** to
           save all related output.
    :param model_family: Which model family your input model belongs to.
           Now ``llama``/``bloom``/``gptneox``/``starcoder`` has been supported.
           If ``model_format='gptq'``, only ``llama`` is supported.
    :param dtype: Which quantized precision will be converted.
           If ``model_format='pth'``, `int4` and `int8` are supported,
           meanwhile `int8` only works for `llama` and `gptneox`.
           If ``model_format='gptq'``, only ``int4`` is supported.
    :param model_format: Specify the model format to be converted. ``pth`` is for
           PyTorch model checkpoint from Hugging Face. ``gptq`` is for GPTQ format
           model from Hugging Face.
    :param lora_id_or_path: The name of the Lora configuration to use. Can be either:
           A string, the model id of a Lora configuration hosted inside a model repo on the 
           Hugging Face Hub.
           A path to a directory a Lora configuration file saved using the save_pretrained 
           method.
    :param **kwargs: Supported keyword arguments includes:

           * ``tmp_path``: Valid when ``model_format='pth'``. It refers to the path
             that stores the intermediate model during the conversion process.
           * ``tokenizer_path``: Valid when ``model_format='gptq'``. It refers to the path
             where ``tokenizer.model`` is located (if it is not in the ``model`` directory)

    :return: the path string to the converted lower precision checkpoint.
    """
    if model_format == "pth":
        with tempfile.TemporaryDirectory() as tmp_merged_dir:
            if lora_id_or_path:
                from bigdl.llm.utils.lora_util import merge_and_export_lora_model
                if "tmp_path" in kwargs:
                    tmp_path = kwargs.get("tmp_path")
                    lora_name = Path(lora_id_or_path).stem if os.path.exists(lora_id_or_path) \
                        else lora_id_or_path.replace("/", "-")
                    merged_dir = os.path.join(tmp_path, f'merged_{lora_name}_{int(time.time())}')
                else:
                    merged_dir = tmp_merged_dir
                print("Merging lora with the base model....")
                model = merge_and_export_lora_model(base_model=model,
                                                    lora_id_or_path=lora_id_or_path,
                                                    output_path=merged_dir)
            _, _used_args = _special_kwarg_check(kwargs=kwargs,
                                                 check_args=["tmp_path"])
            return ggml_convert_model(input_path=model,
                                      output_path=outfile,
                                      model_family=model_family,
                                      dtype=outtype,
                                      **_used_args,
                                      )
    elif model_format == "gptq":
        # TODO: support lora?
        invalidInputError(model_family == "llama" and outtype == 'int4',
                          "Convert GPTQ models should always "
                          "specify `--model-family llama --dtype int4` in the command line.")
        invalidInputError(os.path.isdir(outfile),
                          "The output_path {} is not a directory".format(outfile))
        _, _used_args = _special_kwarg_check(kwargs=kwargs,
                                             check_args=["tokenizer_path"])

        output_filename = "bigdl_llm_{}_{}_from_gptq.bin".format(model_family,
                                                                 outtype.lower())
        outfile = os.path.join(outfile, output_filename)

        if "tokenizer_path" in _used_args:
            gptq_tokenizer_path = _used_args["tokenizer_path"]
        else:
            gptq_tokenizer_path = None

        convert_gptq2ggml(input_path=model,
                          output_path=outfile,
                          tokenizer_path=gptq_tokenizer_path,
                          )
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
    parser.add_argument("--lora-id-or-path", type=str, default=None,
                        help="hugging face lora model name or a path to a *directory* "
                             "containing lora model weights.")

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
