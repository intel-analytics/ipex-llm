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
import torch
from ipex_llm.utils.common import invalidInputError
import time
import sys
from typing import List
from .pipeline_cpp import InitLLMPipeline, generate_serve
from typing import Callable, List, Optional
from transformers import GenerationConfig, \
    LogitsProcessorList, StoppingCriteriaList
import threading
from ipex_llm.utils.common import invalidInputError
import tempfile
import numpy as np
from ipex_llm.transformers.npu_models.lm_head import SlicedLMHead
from multiprocessing import Pool


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
    input_length = np.size(numpy_input)

    new_tokens = new_generate_kwargs['max_new_tokens']
    invalidInputError(input_length + new_tokens <= self.kv_len + 1,
                      "Input plus output tokens should not exceed max_context_len.")
    # TODO: may optimize this part later
    invalidInputError(new_tokens < 1024,
                      f"Generated tokens ({new_tokens}) exceed named pipeline limitation.")

    output_tokens = []

    with tempfile.TemporaryDirectory() as temp_dir:
        # run prefill with PrefillRunner
        output = self(input_ids=inputs,
                      attention_mask=torch.ones(1, inputs.shape[1]).int())
        logits = output.logits
        input_id = torch.argmax(logits[:, -1, :], dim=1)
        input_id.to(torch.int32).numpy().tofile(os.path.join(temp_dir, "input_id.bin"))
        position = np.int64(inputs.shape[1])
        position.tofile(os.path.join(temp_dir, "position.bin"))
        past_key_values = output.past_key_values
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        for layer in range(self.num_layers):
            key_ = key_cache[layer]
            val_ = value_cache[layer]
            new_size = (
                key_.size(0),
                key_.size(1),
                self.kv_len,
                key_.size(3),
            )
            key = key_.as_strided(new_size, key_.stride(), storage_offset=0)
            if not self.transpose_value_cache:
                val = val_.as_strided(new_size, val_.stride(), storage_offset=0)
            else:
                new_size = (
                    val_.size(0),
                    val_.size(1),
                    val_.size(3),
                    self.kv_len,
                )
                val_cache = val_.transpose(-1, -2)
                val = val_cache.as_strided(new_size, val_cache.stride(), storage_offset=0)
            key.to(torch.float16).numpy().tofile(os.path.join(temp_dir, f"key_cache_{layer}.bin"))
            val.to(torch.float16).numpy().tofile(os.path.join(temp_dir, f"value_cache_{layer}.bin"))

        if "eos_token_id" not in new_generate_kwargs:
            eos = 0xffffffff
        else:
            eos = new_generate_kwargs["eos_token_id"]

        time_t1 = time.perf_counter()

        # start generate_serve by Thread
        thread = threading.Thread(target=generate_serve,
                                  args=(self.kv_len, self.num_head,
                                        self.head_dim, self.num_layers,
                                        self.vocab_size,
                                        self.transpose_value_cache,
                                        new_tokens))
        thread.start()

        in_pipe_path = "\\\\.\\pipe\\llminputpipe"
        out_pipe_path = "\\\\.\\pipe\\llmoutputpipe"

        while True:
            try:
                input_pipe = open(in_pipe_path, "wb")
            except:
                time.sleep(1)
            else:
                break

        while True:
            try:
                output_pipe = open(out_pipe_path, "rb")
            except:
                time.sleep(1)
            else:
                break

        time_t2 = time.perf_counter()

        bdata = str.encode(str(temp_dir))
        invalidInputError(len(bdata) <= 2000,
                          f"Length of input directory is too long ({len(bdata)}), "
                          "which may cause read error.")
        input_pipe.write(bdata)
        input_pipe.flush()

        buffersize = 4
        while True:
            data = output_pipe.read(buffersize)
            if len(data) == 0:
                break
            token = int.from_bytes(data, sys.byteorder)
            idx += 1
            if idx == 1:
                time_t3 = time.perf_counter()
            if token == eos:
                break
            output_tokens.append(torch.tensor([token]))
            if streamer is not None:
                streamer.put(torch.tensor([token]))

        output = torch.stack(output_tokens, dim=1)
        output = torch.cat((inputs, output), dim=1)
        if streamer is not None:
            streamer.end()

    thread.join()
    time_end = time.perf_counter()

    self.first_cost = (time_t3 - time_start_all - (time_t2 - time_t1))  # seconds
    self.rest_cost_mean = (time_end - time_t3) / (idx - 1)  # seconds
    self.encoder_time = 0.0

    if do_print:
        print(f" Start the thread and connect the pipe time: {(time_t2 - time_t1):.2f} s")
        print(f" Number of input tokens: {input_length}")
        print(f" Generated tokens: {idx}")
        print(" First token generation time: "
              f"{(time_t3 - time_start_all - (time_t2 - time_t1)):.2f} s")
        print(f" Generation average latency: {(time_end - time_t3) * 1000 /(idx - 1):.2f} ms, "
              f"({(idx - 1)/(time_end - time_t3):.2f} token/s)")
        print(f" Generation time: {(time_end - time_start_all - (time_t2 - time_t1)):.2f} s\n")
    return output


def convert_llm(model: torch.nn.Module,
                kv_len: int,
                max_prompt_len: int,
                transpose_value_cache: bool,
                group_size: int,
                compile_full_model: bool=False,
                save_directory: str=None):
    # whether to set layernorm weight as const
    layernorm_const = os.environ.get("IPEX_LLM_LAYERNORM_CONST", "1") == "1"
    if group_size == 0:
        n_splits_linear = 1
        n_splits_down_proj = 2 if model.config.intermediate_size == 18944 else 1
    else:
        n_splits_linear = model.config.hidden_size // group_size
        n_splits_down_proj = model.config.intermediate_size // group_size
    if model.config.model_type == "llama":
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_dir = os.path.join(temp_dir, "model_weights")
            os.mkdir(weight_dir)
            layer_num = len(model.model.layers)
            from .llama import convert_llama_layer, convert_lm_head_and_embedding
            first_blob_path, last_blob_path = convert_lm_head_and_embedding(model, n_splits_linear,
                                                                            temp_dir, weight_dir)

            param_list = []
            for layer_idx in range(0, layer_num):
                param_list.append((model, layer_idx, n_splits_linear, n_splits_down_proj,
                                   temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                                   layernorm_const))
            with Pool() as pool:
                result = pool.starmap(convert_llama_layer, param_list)

            # Prefill Runner
            from ipex_llm.transformers.npu_models.convert_mp import convert_llama
            convert_llama(model,
                          max_output_len=kv_len,
                          max_prompt_len=max_prompt_len,
                          decoder=False,
                          transpose_value_cache=transpose_value_cache)

            # patch attrs for generate
            model.kv_len = kv_len
            model.num_head = model.model.layers[0].self_attn.num_heads
            model.head_dim = model.model.layers[0].self_attn.head_dim
            model.num_layers = layer_num
            model.transpose_value_cache = transpose_value_cache

            if hasattr(model.model.layers[0].self_attn.rotary_emb, "cos_cached"):
                model_type = "llama"
            else:
                model_type = "llama_32"
            try:
                res = InitLLMPipeline(model_type, kv_len, model.num_head, model.head_dim, layer_num,
                                      model.vocab_size, weight_dir, "model",
                                      first_blob_path, last_blob_path,
                                      os.path.join(temp_dir, "decoder_layer"), layernorm_const)
            except:
                invalidInputError(False,
                                  "False to InitLLMPipeline.")
    elif model.config.model_type == "baichuan":
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_dir = os.path.join(temp_dir, "model_weights")
            os.mkdir(weight_dir)
            layer_num = len(model.model.layers)
            from .baichuan import convert_baichuan_layer, convert_lm_head_and_embedding
            first_blob_path, last_blob_path = convert_lm_head_and_embedding(model, n_splits_linear,
                                                                            temp_dir, weight_dir)

            param_list = []
            for layer_idx in range(0, layer_num):
                param_list.append((model, layer_idx, n_splits_linear, n_splits_down_proj,
                                  temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                                  layernorm_const))
            with Pool() as pool:
                result = pool.starmap(convert_baichuan_layer, param_list)

            # Prefill Runner
            from ipex_llm.transformers.npu_models.convert_mp import convert_baichuan
            convert_baichuan(model,
                             max_output_len=kv_len,
                             max_prompt_len=max_prompt_len,
                             decoder=False,
                             transpose_value_cache=transpose_value_cache)

            # patch attrs for generate
            model.kv_len = kv_len
            model.num_head = model.model.layers[0].self_attn.num_heads
            model.head_dim = model.model.layers[0].self_attn.head_dim
            model.num_layers = layer_num
            model.transpose_value_cache = transpose_value_cache
            model.vocab_size = model.config.vocab_size

            try:
                res = InitLLMPipeline("baichuan", kv_len, model.num_head, model.head_dim, layer_num,
                                      model.vocab_size, weight_dir, "model",
                                      first_blob_path, last_blob_path,
                                      os.path.join(temp_dir, "decoder_layer"), layernorm_const)
            except:
                invalidInputError(False,
                                  "False to InitLLMPipeline.")
    elif model.config.model_type == "minicpm":
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_dir = os.path.join(temp_dir, "model_weights")
            os.mkdir(weight_dir)
            layer_num = len(model.model.layers)
            from .minicpm import convert_minicpm_layer, convert_lm_head_and_embedding
            first_blob_path, last_blob_path = convert_lm_head_and_embedding(model, n_splits_linear,
                                                                            temp_dir, weight_dir)

            param_list = []
            for layer_idx in range(0, layer_num):
                param_list.append((model, layer_idx, n_splits_linear, n_splits_down_proj,
                                   temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                                   layernorm_const))
            with Pool() as pool:
                result = pool.starmap(convert_minicpm_layer, param_list)

            # Prefill Runner
            from ipex_llm.transformers.npu_models.convert_mp import convert_minicpm
            convert_minicpm(model,
                            max_output_len=kv_len,
                            max_prompt_len=max_prompt_len,
                            decoder=False,
                            transpose_value_cache=transpose_value_cache)

            # patch attrs for generate
            model.kv_len = kv_len
            model.num_head = model.model.layers[0].self_attn.num_heads
            model.head_dim = model.model.layers[0].self_attn.head_dim
            model.num_layers = layer_num
            model.transpose_value_cache = transpose_value_cache

            try:
                res = InitLLMPipeline("minicpm", kv_len, model.num_head, model.head_dim, layer_num,
                                      model.vocab_size, weight_dir, "model",
                                      first_blob_path, last_blob_path,
                                      os.path.join(temp_dir, "decoder_layer"), layernorm_const)
            except:
                invalidInputError(False,
                                  "False to InitLLMPipeline.")
    elif model.config.model_type == "qwen2":
        layernorm_const = os.environ.get("IPEX_LLM_LAYERNORM_CONST", "0") == "1"
        with tempfile.TemporaryDirectory() as temp_dir:
            if save_directory is not None:
                temp_dir = save_directory
                os.mkdir(temp_dir)
            weight_dir = os.path.join(temp_dir, "model_weights")
            os.mkdir(weight_dir)
            layer_num = len(model.model.layers)
            from .qwen import convert_qwen_layer, convert_lm_head_and_embedding
            first_blob_path, last_blob_path = convert_lm_head_and_embedding(model, n_splits_linear,
                                                                            temp_dir, weight_dir,
                                                                            compile_full_model)

            param_list = []
            for layer_idx in range(0, layer_num):
                param_list.append((model, layer_idx, n_splits_linear, n_splits_down_proj,
                                  temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                                  layernorm_const))
            with Pool() as pool:
                result = pool.starmap(convert_qwen_layer, param_list)

            if compile_full_model:
                convert_qwen_layer(model, 0, n_splits_linear, n_splits_down_proj,
                                   temp_dir, weight_dir, transpose_value_cache, max_prompt_len,
                                   group_size, layernorm_const, "prefill")

            # Prefill Runner
            from ipex_llm.transformers.npu_models.convert_mp import convert_qwen
            convert_qwen(model,
                         max_output_len=kv_len,
                         max_prompt_len=max_prompt_len,
                         decoder=False,
                         transpose_value_cache=transpose_value_cache)

            # patch attrs for generate
            model.kv_len = kv_len
            model.num_head = model.model.layers[0].self_attn.num_key_value_heads
            model.head_dim = model.model.layers[0].self_attn.head_dim
            model.num_layers = layer_num
            model.transpose_value_cache = transpose_value_cache
            model.vocab_size = model.config.vocab_size

            if save_directory is not None:
                update_dict = {"kv_len": kv_len, "num_head": model.num_head,
                               "head_dim": model.head_dim,
                               "transpose_value_cache": transpose_value_cache,
                               "max_prompt_len": max_prompt_len,
                               "layernorm_const": layernorm_const,
                               "group_size":  group_size}
                model.config.update(update_dict)
                model.config.save_pretrained(save_directory)

            try:
                res = InitLLMPipeline("qwen", kv_len, model.num_head, model.head_dim, layer_num,
                                      model.vocab_size, weight_dir, "model",
                                      first_blob_path, last_blob_path,
                                      os.path.join(temp_dir, "decoder_layer"), layernorm_const)
            except:
                invalidInputError(False,
                                  "False to InitLLMPipeline.")
    else:
        invalidInputError(False, "Now we only support Llama2 / Llama3 / Baichuan2 / "
                                 "Qwen2 / Qwen2.5 / Minicpm for pipeline running.")

    if hasattr(model, "lm_head") and isinstance(model.lm_head, SlicedLMHead):
        model.lm_head.get_fused_lm_head()
    if hasattr(model, "lm_head_1") and isinstance(model.lm_head_1, SlicedLMHead):
        model.lm_head_1.get_fused_lm_head()
        model.lm_head_0.get_fused_lm_head()

    # patch generate function
    import types
    model.generate = types.MethodType(generate, model)
    return model
