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
import asyncio
from transformers import TextIteratorStreamer
logger = logging.get_logger(__name__)


class ModelWorker:
    def __init__(self, checkpoint, low_bit, torch_dtype=torch.float16):
        self.dtype = torch_dtype
        start = time.perf_counter()
        model = self.load_model(checkpoint, low_bit)
        from ipex_llm.utils import BenchmarkWrapper
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
        plain_texts = prompt_request.inputs
        input_ids = None
        inputs_embeds = None
        if "internlm-xcomposer2-vl-7b" in self.model_name.lower():
            lines = [
                "You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).",
                "- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language "
                "model that is developed by Shanghai AI Laboratory (上海人工智能实验室). "
                "It is designed to be helpful, honest, and harmless.",
                "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in "
                "the language chosen by the user such as English and 中文.",
                "- InternLM-XComposer (浦语·灵笔) is capable of comprehending and "
                "articulating responses effectively based on the provided image."
            ]
            meta_instruction = "\n".join(lines)
            if prompt_request.image_list is None:
                inputs = self.model.build_inputs(tokenizer, plain_texts, [], meta_instruction)
                im_mask = torch.zeros(inputs['input_ids'].shape[:2]).bool()
                input_ids = inputs["input_ids"].to('xpu')
            else:
                image = self.model.encode_img(prompt_request.image_list[0])
                plain_texts = "<ImageHere>" + plain_texts
                inputs, im_mask = self.model.interleav_wrap_chat(tokenizer, plain_texts,
                                                                 image, [], meta_instruction)
                inputs_embeds = inputs["inputs_embeds"].to('xpu').to(self.dtype)
        else:
            inputs = tokenizer(plain_texts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to('xpu')
        parameters = prompt_request.parameters
        return input_ids, parameters, request_id, inputs_embeds

    @torch.no_grad()
    async def process_step(self, tokenizer, result_dict):
        if not self.waiting_requests.empty():
            input_ids, parameters, request_id, inputs_embeds = await self.add_request(tokenizer)
            self.streamer[request_id] = TextIteratorStreamer(tokenizer, skip_prompt=True)

            def model_generate():
                generate_kwargs = {k: v for k, v in parameters.dict().items() if v is not None}
                if "codegeex" in self.model_name.lower():
                    eos_token_id = [tokenizer.eos_token_id,
                                    tokenizer.convert_tokens_to_ids("<|user|>"),
                                    tokenizer.convert_tokens_to_ids("<|observation|>")]
                    generate_kwargs["eos_token_id"] = eos_token_id
                elif "internlm-xcomposer2-vl-7b" in self.model_name.lower():
                    eos_token_id = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
                    ]
                    generate_kwargs["eos_token_id"] = eos_token_id
                if input_ids is not None:
                    self.model.generate(input_ids,
                                        streamer=self.streamer[request_id], **generate_kwargs)
                elif inputs_embeds is not None:
                    self.model.generate(inputs_embeds=inputs_embeds,
                                        streamer=self.streamer[request_id], **generate_kwargs)
                torch.xpu.empty_cache()
                torch.xpu.synchronize()

            from threading import Thread
            t1 = Thread(target=model_generate)
            t1.start()
