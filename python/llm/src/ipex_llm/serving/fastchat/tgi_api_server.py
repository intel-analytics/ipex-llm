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

"""
Modified from
https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py

A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server
"""
import asyncio
import argparse
import json
import os
from typing import Generator, Optional, Union, Dict, List, Any

import aiohttp
import fastapi
from fastapi import Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import httpx

try:
    from pydantic.v1 import BaseSettings
except ImportError:
    from pydantic import BaseSettings
import shortuuid
import tiktoken
import uvicorn

from fastchat.constants import (
    WORKER_API_TIMEOUT,
    WORKER_API_EMBEDDING_BATCH_SIZE,
    ErrorCode,
)
from fastchat.conversation import Conversation, SeparatorStyle
from .tgi_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDetails,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    ChatCompletionParam,
    ChatCompletionDetails,
    ChatCompletionResponse,
)
from fastchat.protocol.api_protocol import (
    APIChatCompletionRequest,
    APITokenCheckRequest,
    APITokenCheckResponse,
    APITokenCheckResponseItem,
)
from fastchat.utils import build_logger

logger = build_logger("tgi_api_server", "tgi_api_server.log")

conv_template_map = {}

fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)

async def fetch_remote(url, pload=None, name=None):
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "text": f"{response.reason}",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                return json.dumps(ret)

            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None

    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if request.model not in models:
        ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {request.model}",
        )
    return ret


async def check_length(request, prompt, max_tokens, worker_addr):
    if (
        not isinstance(max_tokens, int) or max_tokens <= 0
    ):  # model worker not support max_tokens=None
        max_tokens = 1024 * 1024

    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    length = min(max_tokens, context_len - token_num)

    if length <= 0:
        return None, create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. However, your messages resulted in {token_num} tokens. Please reduce the length of the messages.",
        )

    return length, None


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.parameters.max_new_tokens is not None and request.parameters.max_new_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.max_new_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.parameters.best_of is not None and request.parameters.best_of <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.best_of} is less than the minimum of 1 - 'n'",
        )
    if request.parameters.temperature is not None and request.parameters.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.parameters.temperature is not None and request.parameters.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.parameters.top_p is not None and request.parameters.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.parameters.top_p is not None and request.parameters.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.top_p} is greater than the maximum of 1 - 'top_p'",
        )
    if request.parameters.top_k is not None and (request.parameters.top_k > -1 and request.parameters.top_k < 1):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )
    if request.parameters.stop is not None and (
        not isinstance(request.parameters.stop, str) and not isinstancerequest.parameters.stop, list
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.parameters.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(text) for text in inp]

    return inp


def create_openai_logprobs(logprob_dict):
    """Create OpenAI-style logprobs."""
    return LogProbs(**logprob_dict) if logprob_dict is not None else None


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


async def get_gen_params(
    model_name: str,
    worker_addr: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
    max_tokens: Optional[int],
    echo: Optional[bool],
    logprobs: Optional[int] = None,
    stop: Optional[Union[str, List[str]]],
    best_of: Optional[int] = None,
    use_beam_search: Optional[bool] = None,
) -> Dict[str, Any]:
    conv = await get_conv(model_name, worker_addr)
    conv = Conversation(
        name=conv["name"],
        system_template=conv["system_template"],
        system_message=conv["system_message"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
        images = []
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.set_system_message(message["content"])
            elif msg_role == "user":
                if type(message["content"]) == list:
                    image_list = [
                        item["image_url"]["url"]
                        for item in message["content"]
                        if item["type"] == "image_url"
                    ]
                    text_list = [
                        item["text"]
                        for item in message["content"]
                        if item["type"] == "text"
                    ]

                    text = "\n".join(text_list)
                    conv.append_message(conv.roles[0], (text, image_list))
                else:
                    conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images = conv.get_images()

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "logprobs": logprobs,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stop_token_ids": conv.stop_token_ids,
    }

    if len(images) > 0:
        gen_params["images"] = images

    if best_of is not None:
        gen_params.update({"best_of": best_of})
    if use_beam_search is not None:
        gen_params.update({"use_beam_search": use_beam_search})

    new_stop = set()
    _add_to_set(stop, new_stop)
    _add_to_set(conv.stop_str, new_stop)

    gen_params["stop"] = list(new_stop)

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    worker_addr = await fetch_remote(
        controller_address + "/get_worker_address", {"model": model_name}, "address"
    )

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str, worker_addr: str):
    conv_template = conv_template_map.get((worker_addr, model_name))
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template

@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    controller_address = app_settings.controller_address
    ret = await fetch_remote(controller_address + "/refresh_all_workers")
    models = await fetch_remote(controller_address + "/list_models", None, "models")
 
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)

async def get_last_model_name_from_list():
    models = await show_available_models()
    return models.data[-1].id

@app.post("/generate", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""

    request.model = await get_last_model_name_from_list()

    worker_addr = await get_worker_address(request.model)

    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.inputs,
        temperature=request.parameters.temperature,
        top_p=request.parameters.top_p,
        top_k=request.parameters.top_k,
        presence_penalty=request.parameters.repetition_penalty,
        frequency_penalty=request.parameters.frequency_penalty,
        max_tokens=request.parameters.max_new_tokens,
        echo=False,
        stop=request.parameters.stop,
    )

    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )

    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

    choices = []
    chat_completions = []
    for i in range(request.parameters.best_of):
        content = asyncio.create_task(generate_completion(gen_params, worker_addr))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if isinstance(content, str):
            content = json.loads(content)

        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
                generated_text=content["text"]
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            choices[-1].generated_tokens = task_usage.dict()["completion_tokens"]
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    
    details = ChatCompletionDetails(
        best_of_sequences=choices
    )
    generated_text = choices[0].message.content
    
    return ChatCompletionResponse(details=details, usage=usage, generated_text=generated_text)


@app.post("/generate_stream", dependencies=[Depends(check_api_key)])
async def create_chat_completion_stream(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""

    request.model = await get_last_model_name_from_list()

    worker_addr = await get_worker_address(request.model)

    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.inputs,
        temperature=request.parameters.temperature,
        top_p=request.parameters.top_p,
        top_k=request.parameters.top_k,
        presence_penalty=request.parameters.repetition_penalty,
        frequency_penalty=request.parameters.frequency_penalty,
        max_tokens=request.parameters.max_new_tokens,
        echo=False,
        stop=request.parameters.stop,
    )

    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )

    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

    generator = chat_completion_stream_generator(
        request.model, gen_params, request.parameters.best_of, worker_addr
    )
    return StreamingResponse(generator, media_type="text/event-stream")


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionStreamChoice(
            index=i,
            message=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        details = ChatCompletionStreamDetails(best_of_sequences=[choice_data])
        
        chunk = ChatCompletionStreamResponse(
            id=id, details=details, generated_text=""
        )
        yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = (
                decoded_unicode
                if len(decoded_unicode) > len(previous_text)
                else previous_text
            )

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionStreamChoice(
                index=i,
                message=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            details = ChatCompletionStreamDetails(best_of_sequences=[choice_data])
            print(f"type of delta_text: {type(delta_text)}")
            chunk = ChatCompletionStreamResponse(
                id=id,
                details=details,
                generated_text=delta_text
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

async def generate_completion_stream(payload: Dict[str, Any], worker_addr: str):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            buffer = b""
            async for raw_chunk in response.aiter_raw():
                buffer += raw_chunk
                while (chunk_end := buffer.find(delimiter)) >= 0:
                    chunk, buffer = buffer[:chunk_end], buffer[chunk_end + 1 :]
                    if not chunk:
                        continue
                    yield json.loads(chunk.decode())


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    return await fetch_remote(worker_addr + "/worker_generate", payload, "")

### END GENERAL API - NOT OPENAI COMPATIBLE ###


def create_openai_api_server():
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.api_keys = args.api_keys

    logger.info(f"args: {args}")
    return args


if __name__ == "__main__":
    args = create_openai_api_server()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")