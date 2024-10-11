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

import time
import numpy
import warnings
import torch
import sys
import transformers
from typing import List
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from .pipeline_cpp import InitLLMPipeline, generate_serve
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import GenerationConfig, \
    LogitsProcessorList, StoppingCriteriaList
import threading
from ipex_llm.utils.common import invalidInputError
import os
from transformers import PretrainedConfig


def patch_flash_attn_import(filename: str) -> List[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def ignore_argument(kwargs: dict, key: "str"):
    arg = kwargs.pop(key, None)
    if arg is not None:
        warnings.warn(f"argument `{key}={arg}` will be ignored")


def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
    # if do_print=True, output timing message
    do_print = kwargs.pop("do_print", False)
    time_start_all, time_t1, idx = time.perf_counter(), None, 0
    new_generate_kwargs = {}
    for var in ['max_new_tokens', 'attention_mask', 'eos_token_id']:
        value = kwargs.pop(var, None)
        if value is not None:
            new_generate_kwargs[var] = value

    if isinstance(inputs[0], torch.Tensor):
        numpy_input = inputs[0].numpy()
    else:
        numpy_input = inputs[0]
    input_length = numpy.size(numpy_input)

    new_tokens = new_generate_kwargs['max_new_tokens']
    invalidInputError(input_length + new_tokens <= self.kv_len + 1,
                      "Input plus output tokens should not exceed max_output_len.")

    # start generate_serve by Thread
    thread = threading.Thread(target=generate_serve,
                              args=(self.kv_len, self.num_head,
                                    self.head_dim, self.num_layers,
                                    new_tokens - 1))
    thread.start()

    in_pipe_path = "\\\\.\\pipe\\llminputpipe"
    out_pipe_path = "\\\\.\\pipe\\llmoutputpipe"

    while True:
        try:
            input_pipe = open(in_pipe_path, "wb")
        except:
            print('Waiting for input pipe')
            time.sleep(1)
        else:
            break

    while True:
        try:
            output_pipe = open(out_pipe_path, "rb")
        except:
            print('Waiting for output pipe')
            time.sleep(1)
        else:
            break

    bdata = b''
    for i in range(0, input_length):
        d = int(numpy_input[i])
        bdata = bdata + d.to_bytes(4, sys.byteorder)

    if "eos_token_id" not in new_generate_kwargs:
        eos = 0xffffffff
    else:
        eos = new_generate_kwargs["eos_token_id"]

    bdata = bdata + eos.to_bytes(4, sys.byteorder)

    time_start = time.perf_counter()

    input_pipe.write(bytearray(bdata))
    input_pipe.flush()

    buffersize = 4
    output_tokens = []
    while True:
        data = output_pipe.read(buffersize)
        if len(data) == 0:
            break
        token = int.from_bytes(data, sys.byteorder)
        idx += 1
        if time_t1 is None:
            time_t1 = time.perf_counter()
        output_tokens.append(torch.tensor([token]))
        if streamer is not None:
            streamer.put(torch.tensor([token]))
        if token == eos:
            break

    output = torch.stack(output_tokens, dim=1)
    output = torch.cat((inputs, output), dim=1)
    if streamer is not None:
        streamer.end()

    thread.join()
    time_end = time.perf_counter()

    if do_print:
        print(f" Start the thread and connect the pipe time: {(time_start - time_start_all):.2f} s")
        print(f" Number of input tokens: {input_length}")
        print(f" Generated tokens: {idx}")
        print(f" First token generation time: {(time_t1 - time_start):.2f} s")
        print(f" Generation average latency: {(time_end - time_t1)*1000 /(idx - 1):.2f} ms, "
              f"({(idx - 1)/(time_end - time_t1):.2f} token/s)")
        print(f" Generation time: {(time_end - time_start):.2f} s\n")

    return output


class NPUModel():
    def __init__(self):
        pass


class _BaseAutoModelClass:
    HF_MODEL = None

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a model from a directory or the HF Hub.
        The loaded model will run supported OPs on NPU, then run other OPs on CPU.

        Three new arguments are added to extend Hugging Face's from_pretrained method as follows:
        :param ov_model: boolean value, whether load blob files from specified directory.
                         If it's False, will convert HF model to specified blob format,
                         but which is not supported now. Default to True.
        :param max_output_len: Maximum context length for whole generation, default to 1024.
        :param model_name: Name prefix of the model weight bin file.
        :return: a model instance
        """
        ov_model = kwargs.get("ov_model", True)
        max_output_len = kwargs.pop("max_output_len", 1024)

        invalidInputError(ov_model,
                          "Original HF model is not supported now.")
        invalidInputError(os.path.exists(pretrained_model_name_or_path),
                          "This directory does not exist, please double check it.")

        config_json = os.path.join(pretrained_model_name_or_path, "config.json")
        invalidInputError(os.path.exists(config_json),
                          "config.json is not found in current directory, please double check it.")
        config = PretrainedConfig.from_json_file(config_json)
        model = NPUModel()
        model.kv_len = max_output_len - 1
        model.num_head = config.num_attention_heads
        model.head_dim = config.hidden_size // config.num_attention_heads
        model.num_layers = config.num_hidden_layers
        model.vocab_size = config.vocab_size

        model_weight_dir = os.path.join(pretrained_model_name_or_path, "model_layer_weights")
        model_name = kwargs.get("model_name", "Model")
        first_blob_name = os.path.join(pretrained_model_name_or_path, "first_model.blob")
        last_blob_name = os.path.join(pretrained_model_name_or_path, "last_model.blob")
        rest_blob_name = os.path.join(pretrained_model_name_or_path, "rest_model.blob")

        for path in [model_weight_dir, first_blob_name, last_blob_name, rest_blob_name]:
            invalidInputError(os.path.exists(path),
                              f"{path} is not found in current directory, please double check it.")

        try:
            res = InitLLMPipeline(model.kv_len, model.num_head, model.head_dim, model.num_layers,
                                  model.vocab_size, model_weight_dir, model_name,
                                  first_blob_name, last_blob_name, rest_blob_name)
        except:
            invalidInputError(False,
                              "False to InitLLMPipeline.")
            exit(0)

        # patch generate function
        import types
        model.generate = types.MethodType(generate, model)
        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSpeechSeq2Seq


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSeq2SeqLM


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSequenceClassification


class AutoModelForMaskedLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMaskedLM


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForQuestionAnswering


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForNextSentencePrediction


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMultipleChoice


class AutoModelForTokenClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForTokenClassification
