from pipeline_models import ModelRunner
import torch.nn.parallel
import torch.distributed as dist
import os
import intel_extension_for_pytorch as ipex

import oneccl_bindings_for_pytorch

from transformers.utils import logging
logger = logging.get_logger(__name__)

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29501'

backend = 'ccl'
dist.init_process_group(backend)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
device = f"xpu:{my_rank}"
logger.info(f"rank: {my_rank}, size: {my_size}")

import time
from transformers import AutoTokenizer, AutoConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio, uuid
from typing import Dict, List, Optional
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
    n_predict: int = 32

empty_req = PromptRequest(prompt="", n_predict=0)

app = FastAPI()
global tokenizer

request_queue: asyncio.Queue = asyncio.Queue()
result_dict: Dict[str, str] = {}
local_rank = my_rank
max_num_seqs = get_int_from_env(["MAX_NUM_SEQS"], "16")


@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    request_id = str(uuid.uuid4())
    await request_queue.put((request_id, prompt_request))
    while True:
        await asyncio.sleep(0.1)
        if request_id in result_dict:
            output_str = result_dict.pop(request_id)
            return {"generated_text": output_str}


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


async def process_requests():
    while True:
        request_ids, prompt_requests = [], []
        for _ in range(max_num_seqs):
            if request_queue.empty():
                break
            request_id, prompt_request = await request_queue.get()
            request_ids.append(request_id)
            prompt_requests.append(prompt_request)

        if local_rank == 0 and prompt_requests:
            # import pdb
            # pdb.set_trace()
            object_list = prompt_requests
            if len(object_list) < max_num_seqs:
                object_list = object_list + [empty_req] * (max_num_seqs - len(object_list))
            logger.info(f"Running: {len(prompt_requests)}, Pending: {request_queue.qsize()}")
            dist.broadcast_object_list(object_list, src=0)
            start_time = time.time()
            outputs = generate_text([req.prompt for req in object_list], [req.n_predict for req in object_list])
            # print(outputs)
            generate_time = time.time() - start_time
            outputs = outputs.cpu()
            output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_strs = output_strs[:len(prompt_requests)]

            for request_id, output_str in zip(request_ids, output_strs):
                result_dict[request_id] = output_str
            # print(result_dict)
            logger.info(f"generate time: {generate_time}")

        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    if local_rank == 0:
        asyncio.create_task(process_requests())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging DeepSpeed-AutoTP')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                    help='The quantization type the model will convert to.')
    parser.add_argument('--port', type=int, default=8000,
                    help='The port number on which the server will run.')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit

    # serialize model initialization so that we do not run out of CPU memory
    for i in range(my_size):
        if my_rank == i:
            logger.info("start model initialization")
            local_model = ModelRunner(model_path, my_rank, my_size, low_bit)
            logger.info("model initialized")
        dist.barrier()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        while True:
            object_list = [None] * max_num_seqs
            dist.broadcast_object_list(object_list, src=0)
            output = generate_text([req.prompt for req in object_list], [req.n_predict for req in object_list])
