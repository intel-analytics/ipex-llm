from torch import nn
import torch
import torch.distributed as dist

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


class DummyLayer(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/models/llama.py#L2076
        self.weight = torch.randn(1,)

    def forward(self, x):
        return x


class Dummy_MLPLayer(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/models/llama.py#L119
        self.up_proj = DummyLayer()

    def forward(self, x):
        return x


class Dummy_DecoderLayer(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/models/llama.py#L2076
        self.input_layernorm = DummyLayer()
        # to avoid AttributeError in https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/models/llama.py#L119
        self.mlp = Dummy_MLPLayer()

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **kwargs):
        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


def load_model(model_path, my_rank, my_size, low_bit='sym_int4'):
    # from llama_models import LlamaForCausalLM
    # if 'llama' in checkpoint.lower():
    #     model = LlamaForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    # return model
    device = f"xpu:{my_rank}"
    from ipex_llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_low_bit=low_bit,
                                                 torch_dtype=torch.float16,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    # print(model)

    nr_slices = my_size
    slice_size = (model.config.num_hidden_layers + nr_slices - 1) // nr_slices
    layer_start = slice_size * my_rank
    layer_end  = layer_start + min(slice_size, model.config.num_hidden_layers - layer_start)

    for i in range(model.config.num_hidden_layers):
        if i < layer_start or i >= layer_end:
            model._modules['model'].layers[i] = Dummy_DecoderLayer()
        else:
            # align layer_idx and len(past_key_values), otherwise abnormal output
            model._modules['model'].layers[i].self_attn.layer_idx = i - layer_start
    if my_rank != 0:
        model._modules['model'].embed_tokens = DummyLayer()
    if my_rank != my_size - 1:
        model._modules['model'].norm = DummyLayer()
        model._modules['lm_head'] = DummyLayer()

    model = model.to(f'xpu:{my_rank}')
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
        model = self.load_model(checkpoint, rank, world_size, low_bit)
        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")
        # from ipex_llm import optimize_model

        # model = optimize_model(model, low_bit=low_bit)
        
        # model = model.to(torch.float16).to(f'xpu:{rank}')
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
                                                    use_cache=True)
        # print(model)

        nr_slices = my_size
        slice_size = (model.config.num_hidden_layers + nr_slices - 1) // nr_slices
        layer_start = slice_size * my_rank
        
        layer_end  = layer_start + min(slice_size, model.config.num_hidden_layers - layer_start)

        for i in range(model.config.num_hidden_layers):
            if i < layer_start or i >= layer_end:
                model._modules['model'].layers[i] = Dummy_DecoderLayer()
            else:
                # align layer_idx and len(past_key_values), otherwise abnormal output
                model._modules['model'].layers[i].self_attn.layer_idx = i - layer_start
        if my_rank != 0:
            model._modules['model'].embed_tokens = DummyLayer()
        if my_rank != my_size - 1:
            model._modules['model'].norm = DummyLayer()
            model._modules['lm_head'] = DummyLayer()

        model = model.to(f'xpu:{my_rank}')
        return model


    def model_step(self, input, cur_batch):
        if cur_batch is None or cur_batch.stopped or input is None:
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
        
        # logger.info(f"{self.rank}, {_past_key_values}")
        output = self.model(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, 
            past_key_values=_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        if False and self.rank > 0:
            _past_key_values = list(output.past_key_values)
            slice_size = (self.model.config.num_hidden_layers + self.world_size - 1) // self.world_size
            layer_start = slice_size * self.rank

            # import pdb
            # pdb.set_trace()
            _past_key_values[0] = [torch.empty_like(output.past_key_values[layer_start][0])]
            _past_key_values = tuple(_past_key_values)
        else:
            _past_key_values = output.past_key_values
        self.past_key_values_dict[cur_id] = _past_key_values
        # import pdb
        # pdb.set_trace()
        if not self.pp_config.is_tail:
            # return output.last_hidden_state
            # print(output.hidden_states[-1].shape)
            return output.hidden_states[-1]
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
                # import pdb
                # pdb.set_trace()
                dist.recv(next_ids, src=self.pre_rank)
                
                if self.tokens.get(cur_id, None) is None:
                    self.tokens[cur_id] = []

                if len(next_ids.shape) == 1:
                    next_ids = next_ids.unsqueeze(0)
                self.tokens[cur_id].append(next_ids)
                self.token_times[cur_id].append(time.perf_counter())
                # self.input_ids_dict[cur_id] += next_ids
                cur_input = next_ids
                # cur_batch.input_len += 1
                cur_batch.input_len = 1
                cur_batch.prompt_lengths = [x + 1 for x in cur_batch.prompt_lengths]

                for index, request_id in enumerate(cur_batch.request_ids):

                    if not self.is_finish.get(request_id, False):
                        remain = cur_batch.max_tokens - len(self.tokens[cur_id])
                        
                        if self.streamer.get(request_id, None) is None:
                            self.streamer[request_id] = asyncio.Queue()
                            
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

                # if self.attention_mask_dict.get(cur_batch.batch_id, None) is None:
                #     self.attention_mask_dict[cur_batch.batch_id] = make_attention_mask(cur_batch.prompt_lengths)
            
        # if self.rank == 0:
        #     logger.info(f"rank: {self.rank}, {batch_list}")
        
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
