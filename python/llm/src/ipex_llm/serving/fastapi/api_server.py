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
from ipex_llm.utils.common import invalidInputError
from transformers.utils import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from ipex_llm.utils.common import invalidInputError
import asyncio
import uuid
from typing import List, Optional, Union, Dict
from .tgi_protocol import Parameters


result_dict: Dict[str, str] = {}
logger = logging.get_logger(__name__)


class InputsRequest(BaseModel):
    inputs: str
    parameters: Optional[Parameters] = None
    stream: Optional[bool] = False
    req_type: str = 'completion'


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


app = FastAPI()
global tokenizer
global local_model


class FastApp():
    def __init__(self, model, mytokenizer):
        global tokenizer
        global local_model
        local_model = model
        tokenizer = mytokenizer
        self.app = app


from .openai_protocol import (
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


def get_queue_next_token(delta_text_queue):
    timeout = int(os.getenv("IPEX_LLM_FASTAPI_TIMEOUT", 60))
    delta_text = delta_text_queue.text_queue.get(timeout=timeout)
    if delta_text is None:
        remain = 0
    else:
        remain = 1
    return delta_text, remain

async def chat_stream_generator(local_model, delta_text_queue, request_id):
    model_name = local_model.model_name
    index = 0
    while True:
        if not hasattr(delta_text_queue, 'empty'):
            delta_text, remain = get_queue_next_token(delta_text_queue)
        else:
            if not delta_text_queue.empty():
                with local_model.dict_lock:
                    remain, delta_text = await delta_text_queue.get()
            else:
                await asyncio.sleep(0)
                continue
        if remain == 0 and delta_text is not None or remain != 0:
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
    local_model.streamer.pop(request_id, None)


async def completion_stream_generator(local_model, delta_text_queue, request_id):
    model_name = local_model.model_name
    index = 0
    while True:
        if not hasattr(delta_text_queue, 'empty'):
            delta_text, remain = get_queue_next_token(delta_text_queue)
        else:
            if not delta_text_queue.empty():
                with local_model.dict_lock:
                    remain, delta_text = await delta_text_queue.get()
            else:
                await asyncio.sleep(0)
                continue
        if remain == 0 and delta_text is not None or remain != 0:
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
                text="",
                logprobs=None,
                finish_reason="length")
            chunk = CompletionStreamResponse(
                id=request_id,
                choices=[choice_data],
                model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            break
    local_model.streamer.pop(request_id, None)


async def generator(local_model, delta_text_queue, request_id):
    while True:
        if not hasattr(delta_text_queue, 'empty'):
            delta_text, remain = get_queue_next_token(delta_text_queue)
            if delta_text is None:
                break
            else:
                yield delta_text
        else:
            if not delta_text_queue.empty():
                with local_model.dict_lock:
                    remain, delta_text = await delta_text_queue.get()
                yield delta_text
                if remain == 0:
                    break
            else:
                await asyncio.sleep(0)
                continue
    local_model.streamer.pop(request_id, None)


@app.post("/generate")
async def generate(inputs_request: InputsRequest):
    if inputs_request.stream:
        result = await generate_stream_api(inputs_request)
        return result
    request_id = str(uuid.uuid4())
    await local_model.waiting_requests.put((request_id, inputs_request))
    while True:
        await asyncio.sleep(0)
        cur_streamer = local_model.streamer.get(request_id, None)
        if cur_streamer is not None:
            output_str = []
            async for item in generator(local_model, cur_streamer, request_id):
                output_str.append(item)
            return request_id, "".join(output_str)


@app.post("/generate_stream")
async def generate_stream_api(inputs_request: InputsRequest):
    request_id, result = await generate_stream(inputs_request)
    return result


async def generate_stream(inputs_request: InputsRequest):
    request_id = str(uuid.uuid4()) + "stream"
    await local_model.waiting_requests.put((request_id, inputs_request))
    while True:
        await asyncio.sleep(0)
        cur_streamer = local_model.streamer.get(request_id, None)
        if cur_streamer is not None:
            if inputs_request.req_type == 'completion':
                cur_generator = completion_stream_generator(local_model, cur_streamer, request_id)
            elif inputs_request.req_type == 'chat':
                cur_generator = chat_stream_generator(local_model, cur_streamer, request_id)
            else:
                invalidInputError(False, "Invalid Request Type.")
            return request_id, StreamingResponse(
                content=cur_generator, media_type="text/event-stream"
            )


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
            invalidInputError(False, f"Unknown role: {role}")
    return prompt.strip()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    model_name = local_model.model_name
    if request.max_tokens is None:
        n_predict = 256
    else:
        n_predict = request.max_tokens
    inputs_request = InputsRequest(
        inputs=get_prompt(request.messages),
        parameters=Parameters(max_new_tokens=n_predict),
        stream=request.stream,
        req_type="chat"
    )
    if request.stream:
        request_id, result = await generate_stream(inputs_request)
    else:
        request_id, result = await generate(inputs_request)
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
        n_predict = 32
    else:
        n_predict = request.max_tokens
    inputs_request = InputsRequest(
        inputs=request.prompt,
        parameters=Parameters(max_new_tokens=n_predict),
        stream=request.stream,
        req_type="completion"
    )
    if request.stream:
        request_id, result = await generate_stream(inputs_request)
    else:
        request_id, result = await generate(inputs_request)
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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests(local_model, result_dict))


async def process_requests(local_model, result_dict):
    while True:
        await asyncio.sleep(0)
        await local_model.process_step(tokenizer, result_dict)
