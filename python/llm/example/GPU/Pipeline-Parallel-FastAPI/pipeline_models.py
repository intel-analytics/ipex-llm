import torch
import torch.distributed as dist

from typing import List, Optional, Tuple, Union, Iterator
import time
from transformers.cache_utils import Cache
from transformers.utils import logging

import numpy as np
import asyncio, uuid
import threading
from pydantic import BaseModel

logger = logging.get_logger(__name__)


class PPConfig:
    """Configuration for ModelSlices."""

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


def make_attention_mask(prompt_lengths):
    max_length = max(prompt_lengths)
    attention_mask = torch.zeros((len(prompt_lengths), max_length), dtype=torch.int64)
    for i, length in enumerate(prompt_lengths):
        attention_mask[i, max_length - length:] = 1
    return attention_mask

class ModelRunner:
    
    def __init__(self, checkpoint, rank, world_size, low_bit, max_num_seqs):

        self.pp_config = PPConfig(rank, world_size)
        
        start = time.perf_counter()
        model = self.load_model(checkpoint, rank, world_size, low_bit)
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
        # self.attention_mask_dict = {}
        self.past_key_values_dict = {}
        self.tokens = {}
        self.token_times = {}
        self.dtype = torch.float16

        self.waiting_requests = asyncio.Queue()
        self.send_buff = None
        self.dict_lock = threading.Lock()

        self.streamer = {}
        self.token_cache = {}
        self.print_len = {}
        self.is_finish = {}
        self.model_name = checkpoint

        self.layer_start = 0


    def load_model(self, model_path, my_rank, my_size, low_bit='sym_int4'):
        device = f"xpu:{my_rank}"
        from ipex_llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_low_bit=low_bit,
                                                    torch_dtype=torch.float16,
                                                    optimize_model=True,
                                                    trust_remote_code=True,
                                                    use_cache=True,
                                                    pipeline_parallel_stages=my_size)
        # print(model)

        # config_class = type(model.config).__name__
        # if config_class == 'ChatGLMConfig':
        #     model.config.num_hidden_layers = model.config.num_layers
        #     nr_slices = my_size
        #     slice_size = (model.config.num_layers + nr_slices - 1) // nr_slices
        #     layer_start = slice_size * my_rank
        #     layer_end  = layer_start + min(slice_size, model.config.num_layers - layer_start)

        #     for i in range(model.config.num_layers):
        #         if i < layer_start or i >= layer_end:
        #             model.transformer.encoder.layers[i] = Dummy_DecoderLayer()
        #         else:
        #             pass
        #             # align layer_idx and len(past_key_values), otherwise abnormal output
        #             # model._modules['encoder'].layers[i].self_attention.layer_idx = i - layer_start
        #             # model.transformer.encoder.layers[i].self_attention.layer_idx = i - layer_start

        #         if my_rank != 0:
        #             model.transformer.embedding = DummyLayer()
        #         if my_rank != my_size - 1:
        #             model.transformer.output_layer = DummyLayer()
                    
        # else:
        #     nr_slices = my_size
        #     slice_size = (model.config.num_hidden_layers + nr_slices - 1) // nr_slices
        #     layer_start = slice_size * my_rank
        #     layer_end  = layer_start + min(slice_size, model.config.num_hidden_layers - layer_start)

        #     for i in range(model.config.num_hidden_layers):
        #         if i < layer_start or i >= layer_end:
        #             model._modules['model'].layers[i] = Dummy_DecoderLayer()
        #         else:
        #             # align layer_idx and len(past_key_values), otherwise abnormal output
        #             model._modules['model'].layers[i].self_attn.layer_idx = i - layer_start
        #     if my_rank != 0:
        #         model._modules['model'].embed_tokens = DummyLayer()
        #     if my_rank != my_size - 1:
        #         model._modules['model'].norm = DummyLayer()
        #         model._modules['lm_head'] = DummyLayer()

        # model = model.to(f'xpu:{my_rank}')
        return model


    def model_step(self, input, cur_batch):
        if cur_batch is None or cur_batch.stopped or input is None:
            return None
        
        cur_id = cur_batch.batch_id
        _past_key_values = self.past_key_values_dict.get(cur_id, None)
        attention_mask = make_attention_mask(cur_batch.prompt_lengths)

        if self.rank == 0:
            input_ids = input
            inputs_embeds = None
        else:
            input_ids = None
            inputs_embeds = input
        
        # logger.info(f"{self.rank}, {_past_key_values}")
        output = self.model(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, 
            past_key_values=_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        use_legacy_cache = not isinstance(output.past_key_values, Cache)
        if use_legacy_cache and self.rank > 0:
            if output.past_key_values[0] is None:
                _past_key_values = list(output.past_key_values)
                slice_size = (self.model.config.num_hidden_layers + self.world_size - 1) // self.world_size
                layer_start = slice_size * self.rank

                _past_key_values[0] = [torch.empty_like(output.past_key_values[layer_start][0])]
                _past_key_values = tuple(_past_key_values)
            else:
                _past_key_values = output.past_key_values
        else:
            _past_key_values = output.past_key_values
        self.past_key_values_dict[cur_id] = _past_key_values
        if not self.pp_config.is_tail:
            return output.hidden_states[-1]
        else:
            return output.logits

    
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

        plain_texts = [req.prompt for req in prompt_requests]
        inputs = tokenizer(plain_texts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(f'xpu:{self.rank}')
        attention_mask = inputs.attention_mask.to(f'xpu:{self.rank}')
        new_batch = BatchTask(
            batch_id="batch_" + str(uuid.uuid4()),
            request_ids=request_ids,
            max_tokens=max([req.n_predict for req in prompt_requests]),
            batch_size=input_ids.size(0),
            input_len=input_ids.size(1),
            prompt_lengths=[sum(attention_mask[i,:]) for i in range(input_ids.size(0))],
            stopped=False,
        )

        self.input_ids_dict[new_batch.batch_id] = input_ids
        self.token_times[new_batch.batch_id] = [time.perf_counter()]

        return new_batch

    
    def clear_batch(self, cur_id):
        self.input_ids_dict.pop(cur_id, None)
        self.tokens.pop(cur_id, None)
        self.token_times.pop(cur_id, None)
        self.past_key_values_dict.pop(cur_id, None)
        # torch.xpu.empty_cache()


    async def process_step(self, tokenizer, result_dict):
        cur_batch = None

        if self.rank == 0:
            if self.send_buff is not None:
                # logger.info(f"rank: {self.rank}, send: {self.send_buff.shape}")
                dist.send(self.send_buff, dst=self.next_rank)

            if self.on_going_batches[0] is not None:
                cur_batch = self.on_going_batches[0]
                cur_input = None
            
            if cur_batch is None:
                if not self.waiting_requests.empty():
                    await asyncio.sleep(0.01)
                    cur_batch = await self.add_request(tokenizer)
                    cur_input = self.input_ids_dict[cur_batch.batch_id]
                else:
                    cur_batch = None
                    cur_input = None

            if (cur_batch is not None) and (not cur_batch.stopped) and (cur_input is None):
                cur_id = cur_batch.batch_id
                next_ids = torch.empty((cur_batch.batch_size, 1,), device=f'xpu:{self.rank}', dtype=torch.int64)
                # logger.info(f"rank: {self.rank}, recv: {next_ids.shape}")
                dist.recv(next_ids, src=self.pre_rank)
                
                if self.tokens.get(cur_id, None) is None:
                    self.tokens[cur_id] = []

                if len(next_ids.shape) == 1:
                    next_ids = next_ids.unsqueeze(0)
                self.tokens[cur_id].append(next_ids)
                self.token_times[cur_id].append(time.perf_counter())
                cur_input = next_ids
                cur_batch.input_len = 1
                cur_batch.prompt_lengths = [x + 1 for x in cur_batch.prompt_lengths]

                for index, request_id in enumerate(cur_batch.request_ids):

                    if not self.is_finish.get(request_id, False):
                        remain = cur_batch.max_tokens - len(self.tokens[cur_id])
                        
                        if self.streamer.get(request_id, None) is None:
                            self.streamer[request_id] = asyncio.Queue()
                            
                        # Currently ignore eos for benchmark
                        # if next_ids[index].int() == tokenizer.eos_token_id:
                        #     remain = 0
                        #     self.is_finish[request_id] = True

                        if self.token_cache.get(request_id, None) is None:
                            self.token_cache[request_id] = []
                            self.print_len[request_id] = 0
                        self.token_cache[request_id].extend(next_ids[index].tolist())

                        text = tokenizer.decode(self.token_cache[request_id])
                        if text.endswith("\n"):
                            printable_text = text[self.print_len[request_id]:]
                            self.token_cache[request_id] = []
                            self.print_len[request_id] = 0
                        elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
                            printable_text = text[self.print_len[request_id]:]
                            self.print_len[request_id] += len(printable_text)
                        else:
                            printable_text = text[self.print_len[request_id] : text.rfind(" ") + 1]
                            self.print_len[request_id] += len(printable_text)

                        if remain > 0:
                            await self.streamer[request_id].put((remain, printable_text))
                        else:
                            printable_text = printable_text + text[self.print_len[request_id]:]
                            self.token_cache.pop(request_id, None)
                            self.print_len.pop(request_id, None)
                            await self.streamer[request_id].put((remain, printable_text))
                
                if len(self.tokens[cur_id]) >= cur_batch.max_tokens:
                    # Finish a batch
                    # logger.info(self.tokens[cur_id])
                    outputs = torch.cat(self.tokens[cur_id], dim=1)
                    outputs = outputs.cpu()
                    output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                    for request_id, output_str in zip(cur_batch.request_ids, output_strs):
                        with self.dict_lock:
                            result_dict[request_id] = output_str

                    cur_times = self.token_times[cur_id]
                    first_token = cur_times[1] - cur_times[0]
                    next_token = (cur_times[-1] - cur_times[1]) / (len(self.tokens[cur_id]) - 1)
                    logger.info(f"First token latency: {first_token}, next token latency: {next_token}")
                    self.clear_batch(cur_id)
                    cur_batch.stopped = True
            else:
                if (cur_batch is not None) and cur_batch.stopped:
                    cur_batch = None

            if cur_batch is not None:
                dist.broadcast_object_list([cur_batch], src=0)
                
        else:
            if self.send_buff is not None:
                # logger.info(f"rank: {self.rank}, send: {self.send_buff.shape}")
                dist.send(self.send_buff, dst=self.next_rank)

            batch_list = [None]
            dist.broadcast_object_list(batch_list, src=0)

            cur_batch = batch_list[0]
            cur_input = None

            if cur_batch is not None:
                if cur_batch.stopped:
                    self.clear_batch(cur_batch.batch_id)
                else:
                    cur_len = cur_batch.input_len
                    cur_input = torch.empty((cur_batch.batch_size, cur_len, self.hidden_size,), device=f'xpu:{self.rank}', dtype=self.dtype)
                    # logger.info(f"rank: {self.rank}, recv: {cur_input.shape}")
                    dist.recv(cur_input, src=self.pre_rank)
        
        output = self.model_step(cur_input, cur_batch)
        if output is not None and self.rank == self.world_size - 1:
            output = torch.argmax(output[:, -1:, :], dim=-1)

        if output is not None:
            # dist.send(output, dst=self.next_rank)
            self.send_buff = output
        else:
            self.send_buff = None
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
