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
from fastapi.middleware.cors import CORSMiddleware
from .tgi_protocol import Parameters
from typing_extensions import Literal
from fastapi import File, UploadFile, Form
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
    TranscriptionRequest,
    TranscriptionResponse,
)

result_dict: Dict[str, str] = {}
logger = logging.get_logger(__name__)


class InputsRequest(BaseModel):
    inputs: str
    parameters: Optional[Parameters] = None
    image_list: Optional[list] = None
    stream: Optional[bool] = False
    req_type: str = 'completion'
    transcription_request:  Optional[TranscriptionRequest] = None


class ChatCompletionRequest(BaseModel):
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    model: str
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    temperature: Optional[float] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    temperature: Optional[float] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

global tokenizer
global local_model
global processor


class FastApp():
    def __init__(self, model, mytokenizer, myprocessor=None):
        global tokenizer
        global local_model
        global processor
        local_model = model
        tokenizer = mytokenizer
        processor = myprocessor
        self.app = app


def get_queue_next_token(delta_text_queue):
    timeout = int(os.getenv("IPEX_LLM_FASTAPI_TIMEOUT", 60))
    delta_text = delta_text_queue.text_queue.get(timeout=timeout)
    if "whisper" in local_model.model_name.lower():
        if delta_text is not None and "<|" in delta_text and "|>" in delta_text:
            import re
            delta_text = re.sub(r'<\|.*?\|>', '', delta_text)
    if delta_text is None:
        remain = 0
    else:
        remain = 1
    return delta_text, remain


def should_return_end_token(next_token):
    if "codegeex" not in local_model.model_name.lower():
        return True
    else:
        if next_token in ["<|user|>", "<|endoftext|>", "<|observation|>"]:
            return False
    return True

async def chat_stream_generator(local_model, delta_text_queue, request_id):
    model_name = local_model.model_name
    index = 0
    while True:
        await asyncio.sleep(0)
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
            if should_return_end_token(delta_text):
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
        await asyncio.sleep(0)
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
            if should_return_end_token(delta_text):
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
        await asyncio.sleep(0.1)
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
    if "codegeex" in local_model.model_name.lower():
        query = messages[-1]["content"]
        if len(messages) <= 1:
            history = []
        else:
            history = [msg for msg in messages[:-1]]
        history.append({"role": "user", "content": query})
        inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False,
                                               return_tensors="pt", return_dict=False)
        return inputs, []
    else:
        prompt = ""
        image_list = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if type(content) == list:
                image_list1 = [
                    item["image_url"]["url"]
                    for item in content
                    if item["type"] == "image_url"
                ]
                image_list.extend(image_list1)
                text_list = [
                    item["text"]
                    for item in content
                    if item["type"] == "text"
                ]
                prompt = "".join(text_list)
            else:
                if role == "system":
                    prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    if "glm" in local_model.model_name.lower():
                        prompt += f"<|user|>\n{content}\n<|assistant|>"
                    else:
                        prompt += f"[INST] {content} [/INST] "
                elif role == "assistant":
                    prompt += f"{content} "
                else:
                    invalidInputError(False, f"Unknown role: {role}")
        return prompt.strip(), image_list


def set_parameters(req):
    if req.max_tokens is None:
        n_predict = 256
    else:
        n_predict = req.max_tokens
    if req.repetition_penalty is not None:
        repetition_penalty = req.repetition_penalty
    elif req.presence_penalty is not None:
        repetition_penalty = req.presence_penalty
    else:
        repetition_penalty = None
    if req.temperature is not None and req.temperature > 1e-4:
        do_sample = True
    else:
        do_sample = False
    return Parameters(max_new_tokens=n_predict, do_sample=do_sample, min_new_tokens=req.min_tokens,
                      top_p=req.top_p, repetition_penalty=repetition_penalty,
                      temperature=req.temperature, top_k=req.top_k)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    model_name = local_model.model_name
    prompt, image_list = get_prompt(request.messages)
    inputs_request = InputsRequest(
        inputs=prompt,
        parameters=set_parameters(request),
        image_list=image_list if len(image_list) >= 1 else None,
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
    inputs_request = InputsRequest(
        inputs=request.prompt,
        parameters=set_parameters(request),
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


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile=File(...),
    model: Optional[str]=Form("default_model"),
    language: Optional[str]=Form("zh"),
    prompt: Optional[str]=Form(None),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]]=Form(None),
    temperature: Optional[float]=Form(None),
    timestamp_granularities: Optional[List[Literal["word", "segment"]]]=Form(None)
):
    file_path = "./" + file.filename
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(await file.read())
    inputs_request = InputsRequest(
        inputs="transcriptions",
        parameters=None,
        stream=False,
        req_type="completion",
        transcription_request=TranscriptionRequest(file=file_path, model=model, language=language)
    )
    request_id, result = await generate(inputs_request)
    rsp = TranscriptionResponse(text=result)
    return rsp


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests(local_model, result_dict))


async def process_requests(local_model, result_dict):
    while True:
        await asyncio.sleep(0)
        await local_model.process_step(tokenizer, result_dict, processor)
