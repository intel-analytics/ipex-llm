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
from ipex_llm.serving.api import FastApp
logger = logging.get_logger(__name__)

class ModelWorker:
    def __init__(self, checkpoint, low_bit, torch_dtype=torch.float16):
        self.dtype = torch_dtype
        start = time.perf_counter()
        model = self.load_model(checkpoint, low_bit)
        from ipex_llm.utils.benchmark_util import BenchmarkWrapper
        self.model = BenchmarkWrapper(model, do_print=True)
        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")
        self.waiting_requests = asyncio.Queue()
        self.streamer = {}
        self.model_name = checkpoint

    def load_model(self, model_path, low_bit='sym_int4'):
        from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         load_in_low_bit=low_bit,
                                                         torch_dtype=self.dtype,
                                                         optimize_model=True,
                                                         trust_remote_code=True,
                                                         use_cache=True,)
        except:
            model = AutoModel.from_pretrained(model_path,
                                              load_in_low_bit=low_bit,
                                              torch_dtype=self.dtype,
                                              optimize_model=True,
                                              trust_remote_code=True,
                                              use_cache=True,)
        model = model.eval().to("xpu")
        return model

    async def add_request(self, tokenizer):
        if self.waiting_requests.empty():
            return
        tmp_result = await self.waiting_requests.get()
        request_id, prompt_request = tmp_result
        plain_texts = prompt_request.prompt
        inputs = tokenizer(plain_texts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to('xpu')
        max_tokens = prompt_request.n_predict
        return input_ids, max_tokens, request_id

    @torch.no_grad()
    async def process_step(self, tokenizer, result_dict):
        if not self.waiting_requests.empty():
            input_ids, max_tokens, request_id = await self.add_request(tokenizer)
            from transformers import TextIteratorStreamer
            self.streamer[request_id] = TextIteratorStreamer(tokenizer, skip_prompt=True)
            def model_generate():
                self.model.generate(input_ids, streamer=self.streamer[request_id], max_new_tokens=max_tokens)
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
            from threading import Thread
            t1 = Thread(target=model_generate)
            t1.start()

async def main():
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

    local_model = ModelWorker(model_path, low_bit)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    myapp = FastApp(local_model, tokenizer)
    config = uvicorn.Config(app=myapp.app, host="0.0.0.0", port=args.port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())