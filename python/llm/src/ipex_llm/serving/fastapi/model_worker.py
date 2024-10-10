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
import os
import time
import asyncio
from PIL import Image
import requests
from transformers import TextIteratorStreamer
logger = logging.get_logger(__name__)


class ModelWorker:
    def __init__(self, checkpoint, low_bit, model_type="normal", torch_dtype=torch.float16):
        self.dtype = torch_dtype
        start = time.perf_counter()
        if model_type == "audio":
            self.model = self.load_model(checkpoint, low_bit, "audio")
        else:
            model = self.load_model(checkpoint, low_bit)
            if "glm-4v" not in checkpoint.lower():
                from ipex_llm.utils import BenchmarkWrapper
                self.model = BenchmarkWrapper(model, do_print=True)
            else:
                # glm-4v-9b does not support benchmark_util now
                self.model = model
        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")
        self.waiting_requests = asyncio.Queue()
        self.streamer = {}
        self.model_name = checkpoint

    def load_model(self, model_path, low_bit='sym_int4', model_type="normal"):
        if model_type == "audio":
            from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path,
                                                              load_in_low_bit=low_bit,
                                                              torch_dtype=self.dtype,
                                                              optimize_model=True,
                                                              trust_remote_code=True,
                                                              use_cache=True)
        else:
            from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
            modules = None
            if "glm-4" in model_path.lower():
                modules = ["encoder.layers.35.mlp", "encoder.layers.36.mlp",
                           "encoder.layers.37.mlp", "encoder.layers.38.mlp",
                           "encoder.layers.39.mlp"]
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path,
                                                             load_in_low_bit=low_bit,
                                                             torch_dtype=self.dtype,
                                                             optimize_model=True,
                                                             trust_remote_code=True,
                                                             modules_to_not_convert=modules,
                                                             use_cache=True,)
            except:
                model = AutoModel.from_pretrained(model_path,
                                                  load_in_low_bit=low_bit,
                                                  torch_dtype=self.dtype,
                                                  optimize_model=True,
                                                  trust_remote_code=True,
                                                  modules_to_not_convert=modules,
                                                  use_cache=True,)
        model = model.eval().to("xpu")
        return model

    def get_local_image_path(self, image_path):
        local_dir = './local_images/'
        local_path = local_dir + os.path.basename(image_path)
        if os.path.exists(image_path) or os.path.exists(local_path):
            pass
        else:
            response = requests.get(image_path)
            if response.status_code == 200:
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                with open(local_path, 'wb') as file:
                    file.write(response.content)
        return local_path

    async def add_asr_request(self, processor):
        if self.waiting_requests.empty():
            return
        tmp_result = await self.waiting_requests.get()
        request_id, request = tmp_result
        transcription_request = request.transcription_request
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=transcription_request.language, task="transcribe")
        audio_path = transcription_request.file
        import librosa
        raw_speech, sampling_rate = librosa.load(audio_path,
                                                 sr=processor.feature_extractor.sampling_rate)
        input_features = processor(
            raw_speech,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        ).input_features.to('xpu')
        return input_features, forced_decoder_ids, request_id

    async def add_request(self, tokenizer):
        if self.waiting_requests.empty():
            return
        tmp_result = await self.waiting_requests.get()
        request_id, prompt_request = tmp_result
        plain_texts = prompt_request.inputs
        input_ids = None
        inputs_embeds = None
        inputs = None
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
                # only process the first image now
                local_path = self.get_local_image_path(prompt_request.image_list[0])
                image = self.model.encode_img(local_path)
                plain_texts = "<ImageHere>" + plain_texts
                inputs, im_mask = self.model.interleav_wrap_chat(tokenizer, plain_texts,
                                                                 image, [], meta_instruction)
                inputs_embeds = inputs["inputs_embeds"].to('xpu').to(self.dtype)
        elif "glm-4v" in self.model_name.lower() and prompt_request.image_list is not None:
            # only process the first image now
            local_path = self.get_local_image_path(prompt_request.image_list[0])
            image = Image.open(local_path)

            inputs = tokenizer.apply_chat_template([{"role": "user", "image": image,
                                                   "content": plain_texts}],
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True)
            inputs = inputs.to('xpu')
        else:
            inputs = tokenizer(plain_texts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to('xpu')
        parameters = prompt_request.parameters
        return input_ids, parameters, request_id, inputs_embeds, inputs

    @torch.no_grad()
    async def process_step(self, tokenizer, result_dict, processor=None):
        if not self.waiting_requests.empty():
            if processor is not None and "whisper" in self.model_name.lower():
                input_features, decoder_ids, request_id = await self.add_asr_request(processor)
                self.streamer[request_id] = TextIteratorStreamer(tokenizer, skip_prompt=True)

                def model_generate():
                    self.model.generate(input_features,
                                        streamer=self.streamer[request_id],
                                        forced_decoder_ids=decoder_ids)
            else:
                input_ids, parameters, request_id, inputs_embeds, inputs = \
                    await self.add_request(tokenizer)
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
                    else:
                        self.model.generate(**inputs,
                                            streamer=self.streamer[request_id], **generate_kwargs)
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
            from threading import Thread
            t1 = Thread(target=model_generate)
            t1.start()
