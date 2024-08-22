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

import torch
from transformers.utils import logging
import time
from transformers import AutoTokenizer
import uvicorn
import asyncio
import argparse
from ipex_llm.serving.fastapi import FastApp
from ipex_llm.serving.fastapi import ModelWorker
logger = logging.get_logger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging ipex-llm')
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

    processor = None
    if "whisper" not in model_path.lower():
        local_model = ModelWorker(model_path, low_bit)
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        local_model = ModelWorker(model_path, low_bit, "audio", torch_dtype=torch.float32)
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
    myapp = FastApp(local_model, tokenizer, processor)
    config = uvicorn.Config(app=myapp.app, host="0.0.0.0", port=args.port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())