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

import torch.distributed as dist
from ipex_llm.transformers import init_pipeline_parallel, PPModelWorker
from ipex_llm.serving.fastapi import FastApp
from transformers.utils import logging
from transformers import AutoTokenizer
import uvicorn
import asyncio
from typing import Dict
import argparse
logger = logging.get_logger(__name__)

init_pipeline_parallel()
my_rank = dist.get_rank()
my_size = dist.get_world_size()
device = f"xpu:{my_rank}"
logger.info(f"rank: {my_rank}, size: {my_size}")
result_dict: Dict[str, str] = {}
local_rank = my_rank

async def main():
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging Pipeline-Parallel')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                        help='The quantization type the model will convert to.')
    parser.add_argument('--port', type=int, default=8000,
                        help='The port number on which the server will run.')
    parser.add_argument('--max-num-seqs', type=int, default=8,
                        help='Max num sequences in a batch.')
    parser.add_argument('--max-prefilled-seqs', type=int, default=0,
                        help='Max num sequences in a batch during prefilling.')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit
    max_num_seqs = args.max_num_seqs
    max_prefilled_seqs = args.max_prefilled_seqs

    # serialize model initialization so that we do not run out of CPU memory
    for i in range(my_size):
        if my_rank == i:
            logger.info("start model initialization")
            local_model = PPModelWorker(model_path, my_rank, my_size, low_bit, max_num_seqs, max_prefilled_seqs)
            logger.info("model initialized")
        dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    myapp = FastApp(local_model, tokenizer)
    if local_rank == 0:
        config = uvicorn.Config(app=myapp.app, host="0.0.0.0", port=args.port)
        server = uvicorn.Server(config)
        await server.serve()
    else:
        while True:
            await asyncio.sleep(0)
            await local_model.process_step(tokenizer, result_dict)

if __name__ == "__main__":
    asyncio.run(main())