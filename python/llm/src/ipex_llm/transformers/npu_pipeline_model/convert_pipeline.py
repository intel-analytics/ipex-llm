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


from openvino.runtime import Core, serialize
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
                      "Input plus output tokens should not exceed max_output_len.")

    # start generate_serve by Thread
    thread = threading.Thread(target=generate_serve,
                              args=(self.kv_len, self.num_head,
                                    self.head_dim, self.num_layers,
                                    self.transpose_value_cache,
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


def update_names_of_IR_and_export_blob(model, model_name, dir):
    xml_path = os.path.join(dir, model_name + ".xml")
    model.save(xml_path)
    new_ir_path = os.path.join(dir, model_name + "_new.xml")
    blob_path = os.path.join(dir, model_name + ".blob")

    core = Core()
    core.set_property("NPU", {"NPU_COMPILATION_MODE_PARAMS":
                              "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add"})
    core.set_property("NPU", {"PERFORMANCE_HINT": "LATENCY"})
    model = core.read_model(xml_path)
    inputs = model.inputs
    for idx, input in enumerate(inputs):
        if len(input.names) == 0:
            model.inputs[idx].set_names({f"input_{idx}"})
    outputs = model.outputs
    for idx, input in enumerate(outputs):
        if len(input.names) == 0:
            model.outputs[idx].set_names({f"output_{idx}"})
    # rewrite this model to a new IR path
    if new_ir_path is not None:
        serialize(model, new_ir_path)

    if blob_path is not None:
        compiledModel = core.compile_model(model, device_name="NPU")
        model_stream = compiledModel.export_model()
        with open(blob_path, 'wb') as f:
            f.write(model_stream)

    os.remove(xml_path)
    os.remove(new_ir_path)

    return blob_path


def convert_llm(model: torch.nn.Module,
                kv_len: int,
                transpose_value_cache: bool):
    if model.config.model_type == "llama":
        from .llama import LowBitLlamaLMHead, LlamaEmbedding
        with tempfile.TemporaryDirectory() as temp_dir:
            # generate lm_head blob
            weight_dir = os.path.join(temp_dir, "model_weights")
            os.mkdir(weight_dir)
            num_heads = model.model.layers[0].self_attn.num_heads
            num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
            head_dim = model.model.layers[0].self_attn.head_dim
            intermediate_size = model.config.intermediate_size
            layer_num = len(model.model.layers)
            rms_norm_eps = model.config.rms_norm_eps
            vocab_size = model.config.vocab_size
            model_norm = model.model.norm
            lm_head = model.lm_head
            weights = [(lm_head.weight, lm_head.scale)]
            if isinstance(weights[0], tuple):
                np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
            else:  # FP16 Linear
                np_dtype = np.float16

            new_lm_head = LowBitLlamaLMHead(
                [1, 1, num_heads * head_dim],
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                max_seq_len=kv_len,
                rms_norm_eps=rms_norm_eps,
                mode="decode",
                transpose_value=False,
                dtype=np_dtype,
                model_norm_weight=model_norm.weight.to(torch.float16),
                vocab_size=vocab_size,
            )
            last_blob_path = update_names_of_IR_and_export_blob(new_lm_head, "lm_head", temp_dir)

            # save weights bins files
            weight_numpy = [
                lm_head.weight.data.numpy(), lm_head.scale.data.numpy(),
            ]

            for idx, weight in enumerate(weight_numpy):
                bin_file = os.path.join(weight_dir, f"model_lm_head_input_{1+idx}.bin")
                weight.tofile(bin_file)

            embedding_layer = model.model.embed_tokens
            new_embedding = LlamaEmbedding(
                vocab_size=model.config.vocab_size,
                embedding_dim=model.config.hidden_size,
                padding_idx=model.config.pad_token_id,
                dtype=np.float16,
            )
            first_blob_path = update_names_of_IR_and_export_blob(new_embedding, "embedding",
                                                                 temp_dir)
            bin_file = os.path.join(weight_dir, f"model_embedding_input_0.bin")
            embedding_layer.weight.to(torch.float16).detach().numpy().tofile(bin_file)

            # generate decoder layer blob
            from ipex_llm.transformers.npu_models.llama_mp import LowBitLlamaMultiDecoderlayer
            for layer_idx in range(0, layer_num):
                curr_layer = model.model.layers[layer_idx]
                attn_layer = curr_layer.self_attn
                mlp_layer = curr_layer.mlp

                weights = []
                for q, k, v, o, g, u, d in zip(attn_layer.q_proj_dq_list,
                                               attn_layer.k_proj_dq_list,
                                               attn_layer.v_proj_dq_list,
                                               attn_layer.o_proj_dq_list,
                                               mlp_layer.gate_proj_dq_list,
                                               mlp_layer.up_proj_dq_list,
                                               mlp_layer.down_proj_dq_list):
                    weights.append((q.weight, q.scale))
                    weights.append((k.weight, k.scale))
                    weights.append((v.weight, v.scale))
                    weights.append((o.weight, o.scale))
                    weights.append((g.weight, g.scale))
                    weights.append((u.weight, u.scale))
                    weights.append((d.weight, d.scale))

                cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
                cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
                layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
                layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

                if isinstance(weights[0], tuple):
                    np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
                else:  # FP16 Linear
                    np_dtype = np.float16

                if layer_idx == 0:
                    single_decoder = LowBitLlamaMultiDecoderlayer(
                        [1, 1, num_heads * head_dim],
                        input_layernorm_weights=None,
                        post_attn_layernorm_weights=None,
                        cached_cos=cached_cos,
                        cached_sin=cached_sin,
                        num_heads=num_heads,
                        num_key_value_heads=num_key_value_heads,
                        num_layers=1,
                        max_seq_len=kv_len,
                        rms_norm_eps=rms_norm_eps,
                        intermediate_size=intermediate_size,
                        mode="decode",
                        transpose_value=transpose_value_cache,
                        dtype=np_dtype,
                    )
                    rest_blob_path = update_names_of_IR_and_export_blob(single_decoder,
                                                                        "decoder_layer",
                                                                        temp_dir)

                input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
                post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
                layer_norm_0.data.numpy().tofile(input_lm_bin_file)
                layer_norm_1.data.numpy().tofile(post_lm_bin_file)

                for idx, (weight, scale) in enumerate(weights):
                    bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{7+idx*2}.bin")
                    weight.numpy().tofile(bin_file)
                    bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{7+idx*2+1}.bin")
                    scale.numpy().tofile(bin_file)

            # patch attrs for generate
            model.kv_len = kv_len
            model.num_head = num_heads
            model.head_dim = head_dim
            model.num_layers = layer_num
            model.transpose_value_cache = transpose_value_cache

            try:
                res = InitLLMPipeline(kv_len, num_heads, head_dim, layer_num,
                                      model.vocab_size, weight_dir, "model",
                                      first_blob_path, last_blob_path, rest_blob_path)
            except:
                invalidInputError(False,
                                  "False to InitLLMPipeline.")
    else:
        invalidInputError(False,
                          "Now we only support Llama2 for pipeline running.")

    # patch generate function
    import types
    model.generate = types.MethodType(generate, model)
    return model
