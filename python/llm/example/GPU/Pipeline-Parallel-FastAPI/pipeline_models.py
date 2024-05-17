from torch import nn
import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex

from typing import List, Optional, Tuple, Union, Iterator
import time
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import numpy as np
import asyncio, uuid
import threading

logger = logging.get_logger(__name__)


class PPConfig:
    """Configuration for ModelSlices."""

    def __init__(self, pp_rank: int, pp_world_size: int) -> None:
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size
        self.is_head = self.pp_rank == 0
        self.is_tail = self.pp_rank == self.pp_world_size - 1

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

class DummyLayer(nn.Module):
    pass


class PipelineBaseModel(nn.Module):
    def __init__(self, config):
        self.pp_config = PPConfig(pp_rank=dist.get_rank(), pp_world_size=dist.get_world_size())
        nr_slices = self.pp_config.pp_world_size
        # self.config.num_hidden_layers = 8
        slice_size = (self.config.num_hidden_layers + nr_slices -
                      1) // nr_slices
        self.layer_start = slice_size * self.pp_config.pp_rank
        self.layer_end  = self.layer_start + min(slice_size,
                                    self.config.num_hidden_layers - self.layer_start)
        self.num_layers = self.layer_end - self.layer_start

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            assert self.pp_config.is_head, "input_ids is only supported on the head stage"
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            assert not self.pp_config.is_head, "inputs_embeds is only supported on the tail stage"
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx in range(self.num_layers):
            decoder_layer = self.layers[self.layer_start + idx]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        if self.pp_config.is_tail:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def load_model(checkpoint):
    from llama_models import LlamaForCausalLM
    if 'llama' in checkpoint.lower():
        model = LlamaForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    return model

from pydantic import BaseModel
class BatchTask(BaseModel):
    batch_id: str
    request_ids: List[str]
    max_tokens: int
    batch_size: int
    input_len: int
    # plain_texts: List[str]
    prompt_lengths: List[int]
    stopped: bool
    # input_ids: torch.Tensor
    # attention_mask: torch.Tensor


def make_attention_mask(prompt_lengths):
    max_length = max(prompt_lengths)
    attention_mask = torch.zeros((len(prompt_lengths), max_length), dtype=torch.int64)
    for i, length in enumerate(prompt_lengths):
        attention_mask[i, max_length - length:] = 1
    return attention_mask

class ModelRunner:
    
    def __init__(self, checkpoint, rank, world_size, low_bit, max_num_seqs):
        
        import sys
        self.pp_config = PPConfig(rank, world_size)
        
        start = time.perf_counter()
        model = load_model(checkpoint)
        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")
        from ipex_llm import optimize_model

        model = optimize_model(model, low_bit=low_bit)
        
        model = model.to(torch.float32).to(f'xpu:{rank}')
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.pre_rank = (self.rank - 1) % self.world_size
        self.next_rank = (self.rank + 1) % self.world_size
        self.hidden_size = self.model.config.hidden_size
    
        self.max_num_seqs = max_num_seqs
        self.batch_list = [None] * self.world_size
        self.on_going_batches = [None] * self.world_size
        self.input_ids_dict = {}
        # self.attention_mask_dict = {}
        self.past_key_values_dict = {}
        self.tokens = {}
        self.token_times = {}

        self.waiting_requests = asyncio.Queue()
        self.send_buff = None
        self.dict_lock = threading.Lock()

                
    # def generate(self, input_ids=None, max_tokens=5, attention_mask=None):
    #     times = []
    #     with torch.no_grad():
    #         _input_ids = None
    #         _past_key_values = None
    #         bs = input_ids.shape[0]
    #         output_ids = input_ids.clone()
    #         for i in range(max_tokens):
    #             start = time.perf_counter()
    #             if _input_ids is None:
    #                 _input_ids = input_ids
    #             if self.rank == 0:
    #                 outputs = self.model(input_ids=_input_ids, attention_mask=attention_mask, past_key_values=_past_key_values, use_cache=True)
    #             else:
    #                 inputs_embeds = torch.empty(_input_ids.shape + (self.hidden_size,) , device=f'xpu:{self.rank}', dtype=torch.float32)
    #                 dist.recv(inputs_embeds, src=self.pre_rank)
    #                 outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=_past_key_values, use_cache=True)
                
    #             if self.rank == self.world_size - 1:
    #                 logits = outputs.logits
    #                 next_ids = torch.argmax(logits[:, -1:, :], dim=-1)
    #                 assert next_ids.shape == (bs, 1)
    #                 dist.broadcast(next_ids, src=self.rank)
    #             else:
    #                 dist.send(outputs.last_hidden_state, dst=self.next_rank)
    #                 next_ids = torch.empty((bs, 1), device=f'xpu:{self.rank}', dtype=torch.int64)
    #                 dist.broadcast(next_ids, src=self.world_size - 1)
                
    #             _input_ids = next_ids
    #             output_ids = torch.cat([output_ids, next_ids], dim=-1)
    #             _past_key_values = outputs.past_key_values
    #             end = time.perf_counter()
    #             times.append(end - start)
        
    #     if self.rank == 0:
    #         logger.info(f"first token latency: {times[0]}, rest token avg latecy: {np.mean(times[1:])}")
    #     return output_ids


    def model_step(self, input):
        # dist.broadcast(batch_list, src=0)
        cur_batch = self.batch_list[self.rank]
        if cur_batch is None or cur_batch.stopped:
            return None
        
        cur_id = cur_batch.batch_id
        _past_key_values = self.past_key_values_dict.get(cur_id, None)
        # attention_mask = self.attention_mask_dict[cur_id]
        attention_mask = make_attention_mask(cur_batch.prompt_lengths)

        if self.rank == 0:
            input_ids = input
            inputs_embeds = None
        else:
            input_ids = None
            inputs_embeds = input
        output = self.model(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, 
            past_key_values=_past_key_values,
            use_cache=True
        )
        self.past_key_values_dict[cur_id] = output.past_key_values
        if not self.pp_config.is_tail:
            return output.last_hidden_state
        else:
            # logger.info(f"logits: {output.logits.shape}")
            return output.logits

    
    def is_initialized(self):
        return True
    
    
    async def add_request(self, tokenizer):
        request_ids, prompt_requests = [], []
        for _ in range(self.max_num_seqs):
            if self.waiting_requests.empty():
                break
            
            tmp_result = await self.waiting_requests.get()
            # logger.info(tmp_result)
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
            # plain_texts=plain_texts,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
        )

        self.input_ids_dict[new_batch.batch_id] = input_ids
        self.token_times[new_batch.batch_id] = [time.perf_counter()]
        # self.attention_mask_dict[new_batch.batch_id] = attention_mask

        return new_batch

    
    def clear_batch(self, cur_id):
        self.input_ids_dict.pop(cur_id, None)
        self.tokens.pop(cur_id, None)
        self.token_times.pop(cur_id, None)
        # self.attention_mask_dict.pop(cur_id, None)
        self.past_key_values_dict.pop(cur_id, None)
        # torch.xpu.empty_cache()


    async def process_step(self, tokenizer, result_dict):
        batch_list = self.batch_list
        cur_batch = None

        if self.rank == 0:
            if self.on_going_batches[0] is not None:
                cur_batch = self.on_going_batches[0]
                cur_input = None
            
            if cur_batch is None:
                if not self.waiting_requests.empty():
                    # await asyncio.sleep(0.01)
                    cur_batch = await self.add_request(tokenizer)
                    cur_input = self.input_ids_dict[cur_batch.batch_id]
                else:
                    cur_batch = None
                    cur_input = None

            batch_list = [cur_batch] + batch_list

            if len(batch_list) < self.world_size:
                batch_list = batch_list + [None] * (self.world_size-len(batch_list))
            batch_list = batch_list[:self.world_size]
            dist.broadcast_object_list(batch_list, src=0)

            if self.send_buff is not None:
                # logger.info(f"rank: {self.rank}, send: {self.send_buff.shape}")
                dist.send(self.send_buff, dst=self.next_rank)

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
                # self.input_ids_dict[cur_id] += next_ids
                cur_input = next_ids
                # batch_list[0].input_len += 1
                batch_list[0].input_len = 1
                batch_list[0].prompt_lengths = [x + 1 for x in batch_list[0].prompt_lengths]
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
                    batch_list[0].stopped = True
            else:
                if (cur_batch is not None) and cur_batch.stopped:
                    batch_list[0] = None
                    cur_batch = None
                
        else:
            batch_list = [None] * self.world_size
            dist.broadcast_object_list(batch_list, src=0)

            cur_batch = batch_list[self.rank]
            cur_input = None
            if cur_batch is not None:
                if cur_batch.stopped:
                    self.clear_batch(cur_batch.batch_id)
                else:
                    cur_len = cur_batch.input_len
                    cur_input = torch.empty((cur_batch.batch_size, cur_len, self.hidden_size,), device=f'xpu:{self.rank}', dtype=torch.float32)
                    # logger.info(f"rank: {self.rank}, recv: {cur_input.shape}")
                    dist.recv(cur_input, src=self.pre_rank)

                # if self.attention_mask_dict.get(cur_batch.batch_id, None) is None:
                #     self.attention_mask_dict[cur_batch.batch_id] = make_attention_mask(cur_batch.prompt_lengths)
            
            if self.send_buff is not None:
                # logger.info(f"rank: {self.rank}, send: {self.send_buff.shape}")
                dist.send(self.send_buff, dst=self.next_rank)
            
        self.batch_list = batch_list
        # if self.rank == 0:
        #     logger.info(f"rank: {self.rank}, {batch_list}")
        
        output = self.model_step(cur_input)
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

