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
# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
#

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
import os
import time
import numpy as np
from typing import Callable, List, Optional, Union, Tuple, Any
from types import SimpleNamespace
import transformers
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ipex_llm.utils.common import invalidInputError
from ipex_llm.ggml.quantize import ggml_tensor_qtype
import logging
logger = logging.getLogger(__name__)
import asyncio
import uuid
import threading
import pickle
try:
    from pydantic import BaseModel
except ImportError:
    from abc import ABCMeta
    BaseModel = ABCMeta

# patch GenerationMixin.generate
from transformers import GenerationMixin
original_generate = GenerationMixin.generate


class DummyLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/
        # python/llm/src/ipex_llm/transformers/models/llama.py#L2076
        self.weight = nn.Parameter(torch.empty(0,), requires_grad=False)

    def forward(self, x):
        return x


class Dummy_MLPLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/
        # python/llm/src/ipex_llm/transformers/models/llama.py#L119
        self.up_proj = DummyLayer()
        self.down_proj = DummyLayer()
        self.shared_expert = SimpleNamespace()
        self.shared_expert.up_proj = DummyLayer()

    def forward(self, x):
        return x


class Dummy_DecoderLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError
        self.input_layernorm = DummyLayer()
        self.mlp = Dummy_MLPLayer()

    def forward(self, hidden_states, *args, **kwargs):
        past_key_value = kwargs.get('past_key_value', None)
        use_cache = kwargs.get('use_cache', False)
        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class Dummy_GLMBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError
        self.input_layernorm = DummyLayer()
        self.mlp = Dummy_MLPLayer()

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        if kv_cache is None:
            return hidden_states, ()
        return hidden_states, kv_cache


def init_pipeline_parallel():
    import oneccl_bindings_for_pytorch
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group('ccl')


def low_mem_convert(model):
    from ipex_llm.transformers.convert import convert_forward
    import importlib
    if 'llama' in model.config.model_type:
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaForCausalLM,
            llama_causallm_forward_4_37_lowmem)
    elif model.config.model_type == "chatglm" and not hasattr(model.config, "vision_config"):
        if model.config.num_layers == 40:
            # for glm4-9b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(
                model,
                module.ChatGLMForConditionalGeneration,
                glm4_conditional_generation_forward_lowmem)
        else:
            # for chatglm3-6b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(
                model,
                module.ChatGLMForConditionalGeneration,
                chatglm3_conditional_generation_forward_lowmem)
    return model


def _check_quantize_kv_cache(model, idx, batch_size):
    # align use_quantize_kv_cache setting for different GPU in pipeline parallel
    pp_quantize_kv_cache = (os.environ.get("BIGDL_QUANTIZE_KV_CACHE", None) == "1") or \
        (os.environ.get("IPEX_LLM_QUANTIZE_KV_CACHE", None) == "1") or \
        (os.environ.get("IPEX_LLM_LOW_MEM", None) == "1")
    if model.config.model_type == "qwen" and hasattr(model.config, "visual"):
        # for Qwen-VL-Chat
        linear = model._modules['transformer'].h[idx].mlp.c_proj
    elif model.config.model_type == "chatglm":
        # for chatglm3-6b, glm-4-9b-chat
        linear = model._modules['transformer'].encoder.layers[idx].self_attention.query_key_value
    else:
        linear = model._modules['model'].layers[idx].mlp.up_proj
    pp_quantize_kv_cache = pp_quantize_kv_cache or (1 < batch_size and batch_size <= 8 and
                                                    hasattr(linear, "qtype") and
                                                    linear.qtype != ggml_tensor_qtype["fp16"] and
                                                    linear.qtype != ggml_tensor_qtype["bf16"])
    if pp_quantize_kv_cache:
        os.environ["IPEX_LLM_QUANTIZE_KV_CACHE"] = "1"
    else:
        os.environ["IPEX_LLM_QUANTIZE_KV_CACHE"] = "0"


def pipeline_parallel(model, pipeline_parallel_stages, torch_dtype=torch.float32):
    global num_layers
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'num_layers'):
        # for chatglm3-6b
        num_layers = model.config.num_layers

    slice_size = (num_layers + pipeline_parallel_stages - 1) // pipeline_parallel_stages

    local_rank = dist.get_rank()

    global layer_start
    global layer_end
    layer_start = slice_size * local_rank
    layer_end = layer_start + min(slice_size, num_layers - layer_start)

    if model.config.model_type == "qwen" and hasattr(model.config, "visual"):
        # for Qwen-VL-Chat
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['transformer'].h[i] = Dummy_DecoderLayer()
        if local_rank != 0:
            model._modules['transformer'].wte = DummyLayer()
            model._modules['transformer'].drop = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['transformer'].ln_f = DummyLayer()
            model._modules['ln_f'] = DummyLayer()
            model._modules['lm_head'] = DummyLayer()
    elif model.config.model_type == "chatglm":
        # for chatglm3-6b, glm-4-9b-chat
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['transformer'].encoder.layers[i] = Dummy_GLMBlock()
            else:
                model._modules['transformer'].encoder.layers[i].self_attention.num_layers = \
                    i - layer_start

        if local_rank != 0:
            model._modules['transformer'].embedding = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['transformer'].encoder.final_layernorm = DummyLayer()
            model._modules['transformer'].output_layer = DummyLayer()
    else:
        for i in range(num_layers):
            if i < layer_start or i >= layer_end:
                model._modules['model'].layers[i] = Dummy_DecoderLayer()
            else:
                model._modules['model'].layers[i].self_attn.layer_idx = i - layer_start

        if local_rank != 0:
            model._modules['model'].embed_tokens = DummyLayer()
        if local_rank != pipeline_parallel_stages - 1:
            model._modules['model'].norm = DummyLayer()
            model._modules['lm_head'] = DummyLayer()

    _enable_lowmem = os.getenv('IPEX_LLM_LOW_MEM')
    _enable_lowmem = (_enable_lowmem is not None) and (_enable_lowmem.lower() == "1")
    if _enable_lowmem:
        model = low_mem_convert(model)

    model.pipeline_parallel_stages = pipeline_parallel_stages
    model.layer_start = layer_start
    model.layer_end = layer_end
    model.num_layers = num_layers
    if torch_dtype == torch.float16:
        model = model.half()
    model = model.to(f'xpu:{local_rank}')
    return model


@torch.no_grad()
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
    if hasattr(self, 'pipeline_parallel_stages') and self.pipeline_parallel_stages > 1:
        # priority: `generation_config` argument > `model.generation_config`
        if generation_config is None:
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning("Setting `pad_token_id` to `eos_token_id`: "
                           f"{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config is not None and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.pop("max_new_tokens")
        else:
            max_new_tokens = kwargs.pop("max_new_tokens", None)

        return self.pipeline_parallel_generate(inputs=inputs,
                                               max_new_tokens=max_new_tokens,
                                               generation_config=generation_config,
                                               **kwargs)

    return original_generate(self,
                             inputs=inputs,
                             generation_config=generation_config,
                             logits_processor=logits_processor,
                             stopping_criteria=stopping_criteria,
                             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                             synced_gpus=synced_gpus,
                             assistant_model=assistant_model,
                             streamer=streamer,
                             **kwargs)

GenerationMixin.generate = generate


@torch.no_grad()
def pipeline_parallel_generate(self,
                               inputs: Optional[torch.Tensor] = None,
                               max_new_tokens: int = 32,
                               generation_config: Optional[GenerationConfig] = None,
                               **kwargs):
    model_kwargs = generation_config.update(**kwargs)
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    bs = inputs_tensor.shape[0]
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=bs,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" \
            else model_kwargs.pop("input_ids")
    local_rank = dist.get_rank()
    pre_rank = (local_rank - 1) % self.pipeline_parallel_stages
    next_rank = (local_rank + 1) % self.pipeline_parallel_stages

    global layer_start
    global layer_end
    global num_layers

    self.first_token_time = 0
    self.next_token_time = []

    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) \
        if eos_token_id is not None else None

    _input_ids = None
    _past_key_values = None

    bs = input_ids.shape[0]
    output_ids = input_ids.clone()
    _check_quantize_kv_cache(self, layer_start, bs)

    step = 0
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    while True:
        if step >= max_new_tokens:
            break

        if _input_ids is None:
            _input_ids = input_ids

        tic = time.time()
        if local_rank == 0:
            outputs = self(input_ids=_input_ids, inputs_embeds=None,
                           past_key_values=_past_key_values, use_cache=True, **model_kwargs)
        else:
            _inputs_shape = _input_ids.shape + (self.config.hidden_size,)
            if step == 0 and self.config.model_type == "chatglm" \
               and hasattr(self.config, "vision_config"):
                # for glm-4v, image features are mapped during 1st token
                # 1597 are computed according to computation process of conv
                _images_feature = 1597 + _input_ids.shape[0] * 2 + _input_ids.shape[1]
                _inputs_shape = (_input_ids.shape[0], _images_feature, self.config.hidden_size,)
            inputs_embeds = torch.empty(_inputs_shape,
                                        device=f'xpu:{local_rank}', dtype=self.dtype)
            dist.recv(inputs_embeds, src=pre_rank)
            outputs = self(input_ids=None, inputs_embeds=inputs_embeds,
                           past_key_values=_past_key_values, use_cache=True, **model_kwargs)

        if local_rank == self.pipeline_parallel_stages - 1:
            logits = outputs.logits
            next_ids = torch.argmax(logits[:, -1:, :], dim=-1)
            dist.broadcast(next_ids, src=local_rank)
        else:
            dist.send(outputs[0].to(self.dtype), dst=next_rank)
            next_ids = torch.empty((bs, 1), device=f'xpu:{local_rank}', dtype=torch.int64)
            dist.broadcast(next_ids, src=self.pipeline_parallel_stages - 1)

        _input_ids = next_ids
        output_ids = torch.cat([output_ids, next_ids], dim=-1)

        # finished sentences should have their next token be a padding token
        next_ids = next_ids.squeeze()
        if eos_token_id is not None:
            if pad_token_id is None:
                invalidInputError(False, "If `eos_token_id` is defined, "
                                         "make sure that `pad_token_id` is defined.")
            next_ids = next_ids * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if self.config.model_type == "chatglm" and self.config.num_layers == 40 \
           and not hasattr(self.config, "vision_config"):
            # for glm-4-9b-chat
            if step == 0:
                value_placeholder = torch.empty_like((outputs.past_key_values)[-1][0])
                past_key_values_placeholder = tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_start)
                ) + (outputs.past_key_values)[: layer_end - layer_start] + tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_end, num_layers)
                )
                _past_key_values = past_key_values_placeholder
            else:
                _past_key_values = outputs.past_key_values
        elif self.config.model_type in ["baichuan", "chatglm"] or \
                (self.config.model_type == "qwen" and hasattr(self.config, "visual")):
            # for baichuan2, chatglm3, Qwen-VL-Chat, glm-4v-9b
            if local_rank != 0:
                value_placeholder = torch.empty_like((outputs.past_key_values)[-1][0])
                past_key_values_placeholder = tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_start)
                ) + (outputs.past_key_values)[layer_start:]
                _past_key_values = past_key_values_placeholder
            else:
                _past_key_values = outputs.past_key_values
        else:
            _past_key_values = outputs.past_key_values

        toc = time.time()
        if step == 0:
            self.first_token_time = toc - tic
        else:
            self.next_token_time.append(toc - tic)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_ids.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        if this_peer_finished:
            break

        step += 1
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
    self.rest_cost_mean = np.mean(self.next_token_time)
    return output_ids


class PPConfig:
    """Configuration for ModelSlices during serving."""
    def __init__(self, pp_rank: int, pp_world_size: int) -> None:
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size
        self.is_head = self.pp_rank == 0
        self.is_tail = self.pp_rank == self.pp_world_size - 1


class BatchTask(BaseModel):
    batch_id: str
    request_ids: List[str]
    max_tokens: int
    batch_size: int
    input_len: int
    prompt_lengths: List[int]
    stopped: bool

    prefilled_index: int
    partial_prefilling: int


def make_attention_mask(prompt_lengths, device):
    max_length = max(prompt_lengths)
    batch_size = len(prompt_lengths)

    range_tensor = torch.arange(max_length, device=device).expand(batch_size, max_length)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    attention_mask = range_tensor >= max_length - prompt_lengths_tensor
    attention_mask = attention_mask.to(torch.int64)

    return attention_mask


class PPModelWorker:
    """Implementation for pipeline parallel multi-stage serving."""
    def __init__(self, checkpoint, rank, world_size, low_bit, max_num_seqs, max_prefilled_seqs,
                 torch_dtype=torch.float16):
        self.pp_config = PPConfig(rank, world_size)
        self.dtype = torch_dtype
        start = time.perf_counter()
        model = self.load_model(checkpoint, world_size, low_bit)
        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")

        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.pre_rank = (self.rank - 1) % self.world_size
        self.next_rank = (self.rank + 1) % self.world_size
        self.hidden_size = self.model.config.hidden_size
        self.max_num_seqs = max_num_seqs
        self.on_going_batches = [None] * self.world_size
        self.input_ids_dict = {}
        self.past_key_values_dict = {}
        self.tokens = {}
        self.token_times = {}
        self.waiting_requests = asyncio.Queue()
        self.send_buff = None
        self.dict_lock = threading.Lock()
        self.streamer = {}
        self.token_cache = {}
        self.print_len = {}
        self.is_finish = {}
        self.model_name = checkpoint

        self.device = f"xpu:{self.rank}"
        # self.layer_start = 0
        # self.layer_end = 0

        self.max_prefilled_seqs = max_prefilled_seqs
        self.partial_output_dict = {}

        self.stream_tasks = {}

    def load_model(self, model_path, world_size, low_bit='sym_int4'):
        from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         load_in_low_bit=low_bit,
                                                         torch_dtype=self.dtype,
                                                         cpu_embedding=True,
                                                         optimize_model=True,
                                                         trust_remote_code=True,
                                                         use_cache=True,
                                                         pipeline_parallel_stages=world_size)
        except:
            model = AutoModel.from_pretrained(model_path,
                                              load_in_low_bit=low_bit,
                                              torch_dtype=self.dtype,
                                              optimize_model=True,
                                              trust_remote_code=True,
                                              use_cache=True,
                                              pipeline_parallel_stages=world_size)
        model = model.eval()
        return model

    def prepare_batch(self, cur_batch):
        if self.rank == 0:
            cur_input_start = cur_batch.prefilled_index
            if self.max_prefilled_seqs > 0:
                if cur_input_start < cur_batch.batch_size:
                    cur_input_end = cur_input_start + self.max_prefilled_seqs
                    cur_input_end = min(cur_input_end, cur_batch.batch_size)
                    cur_batch.partial_prefilling = cur_input_end - cur_input_start
                else:
                    cur_batch.partial_prefilling = 0

        return cur_batch

    def cat_kv_cache(self, model_type, kv_cache_1, kv_cache_2):
        if model_type in ["baichuan", "chatglm", "mixtral"]:
            result = []
            for sub_tuple1, sub_tuple2 in zip(kv_cache_1, kv_cache_2):
                if sub_tuple1 is None:
                    sub_result = [sub_tuple2]
                elif sub_tuple2 is None:
                    sub_result = [sub_tuple1]
                else:
                    sub_result = []
                    for t1, t2 in zip(sub_tuple1, sub_tuple2):
                        if t1 is None:
                            sub_result.append(t2)
                        elif t2 is None:
                            sub_result.append(t1)
                        else:
                            if model_type == "chatglm" and self.model.config.num_layers != 40:
                                sub_result.append(torch.cat((t1, t2), dim=1))
                            else:
                                sub_result.append(torch.cat((t1, t2), dim=0))
                result.append(tuple(sub_result))
            return tuple(result)
        else:
            # num_layers = self.model.layer_end - self.model.layer_start
            num_cache = min(len(kv_cache_1.key_cache), self.model.num_layers)
            for layer_idx in range(num_cache):
                kv_cache_1.key_cache[layer_idx] = \
                    torch.cat([kv_cache_1.key_cache[layer_idx],
                               kv_cache_2.key_cache[layer_idx]], dim=0)
                kv_cache_1.value_cache[layer_idx] = \
                    torch.cat([kv_cache_1.value_cache[layer_idx],
                               kv_cache_2.value_cache[layer_idx]], dim=0)

            return kv_cache_1

    def update_kv_cache(self, kv_cache, prefill=False):
        layer_start = self.model.layer_start
        layer_end = self.model.layer_end
        num_layers = self.model.num_layers

        if self.model.config.model_type == "chatglm" and self.model.config.num_layers == 40:
            # for glm-4-9b-chat
            if prefill:
                value_placeholder = torch.empty_like((kv_cache)[-1][0])
                past_key_values_placeholder = tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_start)
                ) + (kv_cache)[:layer_end - layer_start] + tuple(
                    (value_placeholder, value_placeholder) for _ in range(layer_end, num_layers)
                )
                kv_cache = past_key_values_placeholder
            else:
                pass
        elif self.model.config.model_type in ["baichuan", "chatglm"] and self.rank > 0:
            value_placeholder = torch.empty_like((kv_cache)[-1][0])
            past_key_values_placeholder = tuple(
                (value_placeholder, value_placeholder) for _ in range(layer_start)
            ) + (kv_cache)[layer_start:]
            kv_cache = past_key_values_placeholder
        else:
            pass

        return kv_cache

    @torch.no_grad()
    def model_step(self, input, cur_batch):
        if cur_batch is None or cur_batch.stopped or input is None:
            return None, cur_batch

        # logger.info(f"{self.rank} {cur_batch} {input.shape}")
        cur_id = cur_batch.batch_id
        _past_key_values = self.past_key_values_dict.get(cur_id, None)
        attention_mask = make_attention_mask(cur_batch.prompt_lengths, input.device)

        if self.rank == 0:
            input_ids = input
            inputs_embeds = None

            if cur_batch.partial_prefilling > 0:
                cur_input_start = cur_batch.prefilled_index
                cur_input_end = cur_input_start + cur_batch.partial_prefilling
                input_ids = input_ids[cur_input_start:cur_input_end]
                attention_mask = attention_mask[cur_input_start:cur_input_end]
                tmp_past_key_values = _past_key_values
                _past_key_values = None
        else:
            input_ids = None
            inputs_embeds = input

            if cur_batch.partial_prefilling > 0:
                cur_input_start = cur_batch.prefilled_index
                cur_input_end = cur_input_start + cur_batch.partial_prefilling
                attention_mask = attention_mask[cur_input_start:cur_input_end]
                tmp_past_key_values = _past_key_values
                _past_key_values = None

        torch.xpu.empty_cache()
        output = self.model(input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            past_key_values=_past_key_values,
                            attention_mask=attention_mask,
                            use_cache=True,)

        if cur_batch.partial_prefilling > 0:
            cur_batch.prefilled_index = cur_input_end
            if tmp_past_key_values is None:
                tmp_past_key_values = output.past_key_values
            else:
                tmp_past_key_values = self.cat_kv_cache(self.model.config.model_type,
                                                        tmp_past_key_values,
                                                        output.past_key_values)
                # torch.xpu.empty_cache()

            if cur_batch.prefilled_index == cur_batch.batch_size:
                tmp_past_key_values = self.update_kv_cache(tmp_past_key_values, True)

            self.past_key_values_dict[cur_id] = tmp_past_key_values

            if self.pp_config.is_tail:
                _pre_output = self.partial_output_dict.get(cur_id, None)
                tmp_output = output.logits
                tmp_output = torch.argmax(tmp_output[:, -1:, :], dim=-1)
                if _pre_output is None:
                    _pre_output = tmp_output
                else:
                    _pre_output = torch.cat((_pre_output, tmp_output), dim=0)
                self.partial_output_dict[cur_id] = _pre_output
        else:
            _prefill = self.past_key_values_dict.get(cur_id, None) is None
            _past_key_values = self.update_kv_cache(output.past_key_values, prefill=_prefill)
            self.past_key_values_dict[cur_id] = _past_key_values
        torch.xpu.synchronize()
        if not self.pp_config.is_tail:
            _output = output[0]
            if _output.dtype != self.dtype:
                _output = _output.to(self.dtype)
        else:
            if cur_batch.partial_prefilling > 0 and \
               cur_batch.prefilled_index == cur_batch.batch_size:
                _output = self.partial_output_dict.pop(cur_id, None)
                cur_batch.partial_prefilling = 0
            else:
                _output = torch.argmax(output.logits[:, -1:, :], dim=-1)
        return _output, cur_batch

    def is_initialized(self):
        return True

    async def add_request(self, tokenizer):
        request_ids, prompt_requests = [], []
        for _ in range(self.max_num_seqs):
            if self.waiting_requests.empty():
                break

            tmp_result = await self.waiting_requests.get()
            request_id, prompt_request = tmp_result
            request_ids.append(request_id)
            prompt_requests.append(prompt_request)

        plain_texts = [req.inputs for req in prompt_requests]
        inputs = tokenizer(plain_texts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(f'xpu:{self.rank}')
        attention_mask = inputs.attention_mask.to(f'xpu:{self.rank}')
        new_batch = BatchTask(
            batch_id="batch_" + str(uuid.uuid4()),
            request_ids=request_ids,
            max_tokens=max([req.parameters.max_new_tokens for req in prompt_requests]),
            batch_size=input_ids.size(0),
            input_len=input_ids.size(1),
            prompt_lengths=[sum(attention_mask[i, :]) for i in range(input_ids.size(0))],
            stopped=False,
            prefilled_index=0,
            partial_prefilling=0,
        )

        self.input_ids_dict[new_batch.batch_id] = input_ids
        self.token_times[new_batch.batch_id] = [time.perf_counter()]

        return new_batch

    def clear_batch(self, cur_id):
        self.input_ids_dict.pop(cur_id, None)
        self.tokens.pop(cur_id, None)
        self.token_times.pop(cur_id, None)
        self.past_key_values_dict.pop(cur_id, None)

        self.is_finish.pop(cur_id, None)
        self.partial_output_dict.pop(cur_id, None)

    async def wait_stream_output(self, cur_id):
        cur_task = self.stream_tasks.pop(cur_id, None)
        if cur_task is not None:
            await cur_task

    def get_printable_text(self, cur_text, request_id):
        if cur_text.endswith("\n"):
            printable_text = cur_text[self.print_len[request_id]:]
            self.token_cache[request_id] = []
            self.print_len[request_id] = 0
        elif len(cur_text) > 0 and _is_chinese_char(ord(cur_text[-1])):
            printable_text = cur_text[self.print_len[request_id]:]
            self.print_len[request_id] += len(printable_text)
            self.token_cache[request_id] = []
            self.print_len[request_id] = 0
        else:
            r_index = cur_text.rfind(" ") + 1
            if r_index > self.print_len[request_id]:
                printable_text = cur_text[self.print_len[request_id]: r_index]
                self.token_cache[request_id] = self.token_cache[request_id][-1:]
                self.print_len[request_id] = 0
            else:
                printable_text = cur_text[self.print_len[request_id]: r_index]
        return printable_text

    async def stream_output(self, cur_batch, tokenizer, next_ids):
        cur_id = cur_batch.batch_id
        cur_cached_ids = []
        _stream_tasks = []
        for index, request_id in enumerate(cur_batch.request_ids):
            if not self.is_finish.get(request_id, False):
                if self.token_cache.get(request_id, None) is None:
                    self.token_cache[request_id] = []
                    self.print_len[request_id] = 0
                self.token_cache[request_id].extend(next_ids[index].tolist())
                cur_cached_ids.append(self.token_cache[request_id])

        for index, request_id in enumerate(cur_batch.request_ids):
            if not self.is_finish.get(request_id, False):
                remain = cur_batch.max_tokens - len(self.tokens[cur_id])

                if self.streamer.get(request_id, None) is None:
                    self.streamer[request_id] = asyncio.Queue()

                # Currently ignore eos for benchmark
                # if next_ids[index].int() == tokenizer.eos_token_id:
                #     remain = 0
                #     self.is_finish[request_id] = True

                cur_text = tokenizer.decode(self.token_cache[request_id])
                printable_text = self.get_printable_text(cur_text, request_id)

                if remain > 0:
                    _stream_tasks.append(self.streamer[request_id].put((remain, printable_text)))
                else:
                    printable_text = printable_text + cur_text[self.print_len[request_id]:]
                    self.token_cache.pop(request_id, None)
                    self.print_len.pop(request_id, None)
                    _stream_tasks.append(self.streamer[request_id].put((remain, printable_text)))
        await asyncio.gather(*_stream_tasks)

    async def process_step(self, tokenizer, result_dict, processor=None):
        cur_batch = None
        torch.xpu.synchronize(self.device)
        if self.rank == 0:
            if self.on_going_batches[0] is not None:
                cur_batch = self.on_going_batches[0]
                cur_input = None

            if cur_batch is None:
                if not self.waiting_requests.empty():
                    # wait more requests to be put in self.waiting_requests
                    await asyncio.sleep(0.01)
                    cur_batch = await self.add_request(tokenizer)
                    cur_input = self.input_ids_dict[cur_batch.batch_id]
                else:
                    cur_batch = None
                    cur_input = None

            if (cur_batch is not None) and (not cur_batch.stopped) and (cur_input is None):
                cur_id = cur_batch.batch_id
                if cur_batch.prefilled_index >= cur_batch.batch_size:
                    cur_batch.partial_prefilling = 0
                if cur_batch.partial_prefilling > 0:
                    next_ids = torch.empty((cur_batch.partial_prefilling, 1,),
                                           device=f'xpu:{self.rank}', dtype=torch.int64)
                else:
                    next_ids = torch.empty((cur_batch.batch_size, 1,),
                                           device=f'xpu:{self.rank}', dtype=torch.int64)

                # logger.info(f"recv {self.rank} {next_ids.shape}")
                dist.recv(next_ids, src=self.pre_rank)
                torch.xpu.synchronize(self.device)

                if cur_batch.partial_prefilling > 0:
                    cur_input = self.input_ids_dict[cur_batch.batch_id]
                else:
                    if self.tokens.get(cur_id, None) is None:
                        self.tokens[cur_id] = []
                    if len(next_ids.shape) == 1:
                        next_ids = next_ids.unsqueeze(0)
                    self.tokens[cur_id].append(next_ids)
                    self.token_times[cur_id].append(time.perf_counter())
                    cur_input = next_ids
                    cur_batch.input_len = 1
                    cur_batch.prompt_lengths = [x + 1 for x in cur_batch.prompt_lengths]

                    pre_task = self.stream_tasks.get(cur_id)
                    if pre_task is not None:
                        await pre_task
                        del self.stream_tasks[cur_id]
                    cur_task = asyncio.create_task(
                        self.stream_output(cur_batch, tokenizer, next_ids)
                    )
                    self.stream_tasks[cur_id] = cur_task

                    if len(self.tokens[cur_id]) >= cur_batch.max_tokens:
                        # Finish a batch
                        outputs = torch.cat(self.tokens[cur_id], dim=1)
                        outputs = outputs.cpu()
                        output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                        for request_id, output_str in zip(cur_batch.request_ids, output_strs):
                            with self.dict_lock:
                                result_dict[request_id] = output_str

                        cur_times = self.token_times[cur_id]
                        first_token = cur_times[1] - cur_times[0]
                        next_token = (cur_times[-1] - cur_times[1]) / (len(self.tokens[cur_id]) - 1)
                        logger.info(f"First token latency: {first_token}, "
                                    f"next token latency: {next_token}")
                        await self.wait_stream_output(cur_id)
                        self.clear_batch(cur_id)
                        cur_batch.stopped = True
            else:
                if (cur_batch is not None) and cur_batch.stopped:
                    cur_batch = None

            if cur_batch is not None:
                cur_batch = self.prepare_batch(cur_batch)
                dist.broadcast_object_list([cur_batch], src=0)
            else:
                await asyncio.sleep(0)

        else:
            batch_list = [None]
            dist.broadcast_object_list(batch_list, src=0)
            cur_batch = batch_list[0]
            cur_input = None

            if cur_batch is not None:
                if cur_batch.stopped:
                    self.clear_batch(cur_batch.batch_id)
                else:
                    cur_batch = self.prepare_batch(cur_batch)
                    cur_len = cur_batch.input_len
                    if cur_batch.partial_prefilling:
                        cur_input = torch.empty(
                            (cur_batch.partial_prefilling, cur_len, self.hidden_size,),
                            device=f'xpu:{self.rank}',
                            dtype=self.dtype,
                        )
                    else:
                        cur_input = torch.empty(
                            (cur_batch.batch_size, cur_len, self.hidden_size,),
                            device=f'xpu:{self.rank}',
                            dtype=self.dtype,
                        )
                    # logger.info(f"recv {self.rank} {cur_input.shape}")
                    dist.recv(cur_input, src=self.pre_rank)
                    torch.xpu.synchronize(self.device)

        output, cur_batch = self.model_step(cur_input, cur_batch)

        torch.xpu.synchronize(self.device)
        if self.send_buff is not None:
            self.send_buff.wait()
        if output is not None:
            self.send_buff = dist.isend(output, dst=self.next_rank)

        if self.rank == 0:
            self.on_going_batches[:-1] = self.on_going_batches[1:]
            self.on_going_batches[self.world_size - 1] = cur_batch


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def llama_causallm_forward_4_37_lowmem(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    # ipex-llm change starts

    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)  # noqa
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]  # noqa
        logits = torch.cat(logits, dim=-1)
    else:
        # Only empty cache for first token
        if hidden_states.shape[1] > 1:
            torch.xpu.empty_cache()
        logits = self.lm_head(hidden_states)
        # Only empty cache for first token
        if hidden_states.shape[1] > 1:
            torch.xpu.empty_cache()
    # logits = logits.float()

    # ipex-llm change ends

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def chatglm3_conditional_generation_forward_lowmem(
    self,
    input_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[-1:]

    # ipex-llm change starts
    torch.xpu.empty_cache()
    lm_logits = self.transformer.output_layer(hidden_states)
    torch.xpu.empty_cache()
    lm_logits = lm_logits.transpose(0, 1).contiguous()

    loss = None
    if labels is not None:
        # lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
    # ipex-llm change ends

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def glm4_conditional_generation_forward_lowmem(
    self,
    input_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[:, -1:]
    # ipex-llm change starts
    torch.xpu.empty_cache()
    lm_logits = self.transformer.output_layer(hidden_states)
    torch.xpu.empty_cache()

    loss = None
    if labels is not None:
        # lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
    # ipex-llm change ends

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
