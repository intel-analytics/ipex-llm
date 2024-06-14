from pipeline_models import ModelRunner
import torch.nn.parallel
import torch.distributed as dist
import os

import ipex_llm
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers import init_pipeline_parallel
import oneccl_bindings_for_pytorch
import json

from transformers.utils import logging
logger = logging.get_logger(__name__)

init_pipeline_parallel()

my_rank = dist.get_rank()
my_size = dist.get_world_size()
device = f"xpu:{my_rank}"
logger.info(f"rank: {my_rank}, size: {my_size}")

import time
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio, uuid
from typing import Dict, List, Optional, Any, Callable, Union
import argparse

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)


class PromptRequest(BaseModel):
    prompt: str
    n_predict: Optional[int] = 256
    req_type: str = 'completion'

from openai.types.chat import ChatCompletionMessageParam
class ChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

empty_req = PromptRequest(prompt="", n_predict=0)

app = FastAPI()
global tokenizer
global local_model

request_queue: asyncio.Queue = asyncio.Queue()
result_dict: Dict[str, str] = {}
streamer_dict = {}
local_rank = my_rank


from openai_protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    CompletionResponseChoice,
    CompletionResponse,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)


async def chat_stream_generator(local_model, delta_text_queue, request_id):
    model_name = local_model.model_name
    index = 0
    while True:
        if not delta_text_queue.empty():
            with local_model.dict_lock:
                remain, delta_text = await delta_text_queue.get()
            # print(remain)
            choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(role="assistant", content=delta_text),
                            logprobs=None,
                            finish_reason=None)
            chunk = ChatCompletionStreamResponse( 
                            id=request_id,
                            choices=[choice_data],
                            model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            index = index + 1
            if remain == 0:
                choice_data = ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(role="assistant", content=None),
                                logprobs=None,
                                finish_reason="length")
                chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                choices=[choice_data],
                                model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
                break
        else:
            await asyncio.sleep(0)
    local_model.streamer.pop(request_id, None)


async def completion_stream_generator(local_model, delta_text_queue, request_id):
    model_name = local_model.model_name
    index = 0
    while True:
        if not delta_text_queue.empty():
            with local_model.dict_lock:
                remain, delta_text = await delta_text_queue.get()
            # print(remain)
            choice_data = CompletionResponseStreamChoice(
                            index=index,
                            text=delta_text,
                            logprobs=None,
                            finish_reason=None)
            chunk = CompletionStreamResponse(
                            id=request_id,
                            choices=[choice_data],
                            model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            index = index + 1
            if remain == 0:
                choice_data = CompletionResponseStreamChoice(
                                index=index,
                                text=None,
                                logprobs=None,
                                finish_reason="length")
                chunk = CompletionStreamResponse(
                                id=request_id,
                                choices=[choice_data],
                                model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
                break
        else:
            await asyncio.sleep(0)
    local_model.streamer.pop(request_id, None)


async def generator(local_model, delta_text_queue, request_id):
    while True:
        if not delta_text_queue.empty():
            with local_model.dict_lock:
                remain, delta_text = await delta_text_queue.get()
            yield delta_text
            if remain == 0:
                break
        else:
            await asyncio.sleep(0)
    # streamer_dict.pop(request_id, None)
    local_model.streamer.pop(request_id, None)


@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    request_id = str(uuid.uuid4())
    await local_model.waiting_requests.put((request_id, prompt_request))
    while True:
        await asyncio.sleep(0)
        cur_streamer = local_model.streamer.get(request_id, None)
        if cur_streamer is not None:
            output_str = []
            async for item in generator(local_model, cur_streamer, request_id):
                output_str.append(item)
            return request_id, "".join(output_str)


@app.post("/generate_stream/")
async def generate_stream(prompt_request: PromptRequest):
    request_id = str(uuid.uuid4()) + "stream"
    await local_model.waiting_requests.put((request_id, prompt_request))
    while True:
        await asyncio.sleep(0)
        cur_streamer = local_model.streamer.get(request_id, None)
        if cur_streamer is not None:
            if prompt_request.req_type == 'completion':
                cur_generator = completion_stream_generator(local_model, cur_streamer, request_id)
            elif prompt_request.req_type == 'chat':
                cur_generator = chat_stream_generator(local_model, cur_streamer, request_id)
            else:
                invalidInputError(False, "Invalid Request Type.")

            return request_id, StreamingResponse(
                content=cur_generator, media_type="text/event-stream"
            )


DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(messages) -> str:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
        elif role == "user":
            prompt += f"[INST] {content} [/INST] "
        elif role == "assistant":
            prompt += f"{content} "
        else:
            raise ValueError(f"Unknown role: {role}")
    return prompt.strip()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    model_name = local_model.model_name
    if request.max_tokens is None:
        n_predict = 256
    else:
        n_predict = request.max_tokens
    prompt_request = PromptRequest(
        prompt=get_prompt(request.messages),
        n_predict=n_predict,
        req_type="chat"
    )
    if request.stream:
        request_id, result = await generate_stream(prompt_request)
    else:
        request_id, result = await generate(prompt_request)
        choice_data = ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result),
                        logprobs=None,
                        finish_reason="length")
        result = ChatCompletionResponse( 
                        id=request_id,
                        choices=[choice_data],
                        model=model_name)
    return result

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    model_name = local_model.model_name
    if request.max_tokens is None:
        n_predict = 256
    else:
        n_predict = request.max_tokens
    prompt_request = PromptRequest(
        prompt=request.prompt,
        n_predict=n_predict,
        req_type="completion"
    )
    if request.stream:
        request_id, result = await generate_stream(prompt_request)
    else:
        request_id, result = await generate(prompt_request)
        choice_data = CompletionResponseChoice(
                            index=0,
                            text=result,
                            logprobs=None,
                            finish_reason="length")
        result = CompletionResponse(
                            id=request_id,
                            choices=[choice_data],
                            model=model_name)
    return result


def generate_text(prompt: List[str], n_predict = 32):
    while prompt[-1] == "":
        prompt = prompt[:-1]
    if isinstance(n_predict, list):
        n_predict = max(n_predict)
        
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(f'xpu:{local_rank}')
    print(inputs)
    attention_mask = inputs.attention_mask.to(f'xpu:{local_rank}')
    output = local_model.generate(input_ids,
                                  max_tokens=n_predict,
                            # attention_mask=attention_mask,
                            # max_new_tokens=n_predict,
                            # min_new_tokens=n_predict,
                            # do_sample=False,
                            # use_cache=True
                            )
    torch.xpu.synchronize()

    return output


async def process_requests(local_model, result_dict):
    while True:
        await asyncio.sleep(0)
        await local_model.process_step(tokenizer, result_dict)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests(local_model, result_dict))

async def main():
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging DeepSpeed-AutoTP')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                    help='The quantization type the model will convert to.')
    parser.add_argument('--port', type=int, default=8000,
                    help='The port number on which the server will run.')
    parser.add_argument('--max-num-seqs', type=int, default=8,
                    help='Max num sequences in a batch.')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit
    max_num_seqs = args.max_num_seqs

    # serialize model initialization so that we do not run out of CPU memory
    for i in range(my_size):
        if my_rank == i:
            logger.info("start model initialization")
            global local_model
            local_model = ModelRunner(model_path, my_rank, my_size, low_bit, max_num_seqs)
            logger.info("model initialized")
        dist.barrier()
    # Load tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        config = uvicorn.Config(app=app, host="0.0.0.0", port=args.port)
        server = uvicorn.Server(config)
        await server.serve()
    else:
        while True:
            await asyncio.sleep(0)
            await local_model.process_step(tokenizer, result_dict)

if __name__ == "__main__":
    asyncio.run(main())