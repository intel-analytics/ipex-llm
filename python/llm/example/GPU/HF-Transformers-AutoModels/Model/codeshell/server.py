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

import time
import json
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, List
from threading import Thread
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM, AutoModel
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
from transformers.generation import GenerationConfig, TextIteratorStreamer
from transformers import StoppingCriteriaList, StoppingCriteria
from sse_starlette.sse import EventSourceResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationParameters(BaseModel):
    max_new_tokens: int
    temperature: float
    repetition_penalty: float
    top_p: float
    do_sample: bool
    stop: List[str]


class GenerationRequest(BaseModel):
    inputs: str
    parameters: GenerationParameters


class StopWordsCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, input_length, stop_words, tokenizer):
        self.input_length = input_length
        self.stop_words = stop_words
        self.stop_words += ["|<end|", "|end>|"]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        texts =  [ self.tokenizer.decode(ids[self.input_length:]) for ids in input_ids ]
        dones = [ any(stop_word in text for stop_word in self.stop_words) for text in texts ]
        return all(dones)


@app.post("/generate")
async def generate(request: GenerationRequest):
    global model, tokenizer, device, max_context

    if device == 'xpu':
        torch.xpu.empty_cache()

    prompt = request.inputs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = len(input_ids[0])

    if input_length > max_context:
        tokens = list(input_ids[0])
        prefix_index = tokens.index(70001)   # fim_prefix
        middle_index = tokens.index(70002)   # fim_middle
        suffix_index = tokens.index(70003)   # fim_suffix

        prefix_tokens = tokens[prefix_index+1:suffix_index]
        suffix_tokens = tokens[suffix_index+1:middle_index]
        prefix_len = suffix_index - prefix_index - 1
        suffix_len = middle_index - suffix_index - 1

        if prefix_len + suffix_len > max_context:
            new_prefix_len = max_context * prefix_len // (prefix_len + suffix_len)
            new_suffix_len = max_context * suffix_len // (prefix_len + suffix_len)
            new_prefix_tokens = prefix_tokens[-new_prefix_len:]
            new_suffix_tokens = suffix_tokens[:new_suffix_len]

            input_ids = torch.tensor(
                tokens[:prefix_index+1] +
                new_prefix_tokens +
                tokens[suffix_index:suffix_index+1] +
                new_suffix_tokens +
                tokens[middle_index:]
            ).reshape(1, -1)
            input_length = len(input_ids[0])
            prompt = tokenizer.decode(input_ids[0])

    input_ids = input_ids.to(device)

    stopping_criteria = StoppingCriteriaList(
        [ StopWordsCriteria(input_length, request.parameters.stop, tokenizer) ]
    )

    generation_kwargs = dict(stopping_criteria=stopping_criteria,
                             max_new_tokens=request.parameters.max_new_tokens,
                             temperature=request.parameters.temperature,
                             repetition_penalty=request.parameters.repetition_penalty,
                             top_p=request.parameters.top_p,
                             do_sample=request.parameters.do_sample)

    print('-'*80)
    print('input prompt:', prompt)
    print('input length:', input_length)
    print('-'*80)

    output_ids = model.generate(input_ids, **generation_kwargs)
    output_text = tokenizer.decode(output_ids[0])

    return JSONResponse({
                "generated_text": output_text[len(prompt):]
            })


@app.post("/generate_stream")
async def generate_stream(request: GenerationRequest):
    global model, tokenizer, device, multi_turn

    if device == 'xpu':
        torch.xpu.empty_cache()

    prompt = request.inputs

    if multi_turn:
        prompt = prompt
    else:
        # extract the last turn input
        human_ins = "## human"
        first_ins = prompt.find(human_ins)
        last_ins = prompt.rfind(human_ins)
        prompt = prompt[:first_ins] + prompt[last_ins:]

    input_ids = tokenizer(prompt, return_tensors="pt")
    input_length = len(input_ids['input_ids'][0])
    input_ids = input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    stopping_criteria = StoppingCriteriaList(
        [ StopWordsCriteria(input_length, request.parameters.stop, tokenizer) ]
    )

    max_batch = 1024
    if input_length <= max_batch:
        past_key_values = None
    else:
        with torch.inference_mode():
            past_key_values = None
            for start_pos in range(0, input_length - 1, max_batch):
                end_pos = min(start_pos + max_batch, input_length - 1)
                output = model.forward(input_ids['input_ids'][:, start_pos:end_pos],
                                       past_key_values=past_key_values)
                past_key_values = output.past_key_values

    generation_kwargs = dict(input_ids,
                             past_key_values=past_key_values,
                             streamer=streamer,
                             stopping_criteria=stopping_criteria,
                             max_new_tokens=request.parameters.max_new_tokens,
                             temperature=request.parameters.temperature,
                             repetition_penalty=request.parameters.repetition_penalty,
                             top_p=request.parameters.top_p,
                             do_sample=request.parameters.do_sample)

    print('-'*80)
    print('input prompt:', prompt)
    print('input length:', input_length)
    print('-'*80)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def create_response(streamer):
        for word in tqdm(streamer, "Generating Tokens", unit="token"):
            yield json.dumps({
                "token": {
                    "id": 0,
                    "text": word,
                },
            })

    return EventSourceResponse(create_response(streamer), media_type="text/event-stream")


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="CodeShell-7B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device name."
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8080,
        help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Enable multi-turn chat",
    )
    parser.add_argument(
        "--cpu-embedding",
        action="store_true",
        help="Move Embedding layer to CPU"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=300,
        help="Max context length when using code completion",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        load_in_4bit=True,
        cpu_embedding=args.cpu_embedding
    ).eval()

    device = args.device
    multi_turn = args.multi_turn
    max_context = args.max_context

    model = model.to(device)

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
