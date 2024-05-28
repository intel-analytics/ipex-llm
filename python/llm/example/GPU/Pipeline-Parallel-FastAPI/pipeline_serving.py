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
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer
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
global local_model

request_queue: asyncio.Queue = asyncio.Queue()
result_dict: Dict[str, str] = {}
local_rank = my_rank
max_num_seqs = get_int_from_env(["MAX_NUM_SEQS"], "16")


@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    request_id = str(uuid.uuid4())
    await local_model.waiting_requests.put((request_id, prompt_request))
    while True:
        if request_id in result_dict:
            with local_model.dict_lock:
                output_str = result_dict[request_id]
            if len(output_str) == 0:
                logger.info(f"Why? {request_id}")
                # await asyncio.sleep(0.1)
                # continue
            result_dict.pop(request_id)
            return {"generated_text": output_str}
        await asyncio.sleep(0)            


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