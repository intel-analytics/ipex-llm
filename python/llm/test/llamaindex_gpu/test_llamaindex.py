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
import pytest
from unittest import TestCase
import os
from ipex_llm.llamaindex.llms import IpexLLM

class Test_LlamaIndex_Transformers_API(TestCase):
    def setUp(self):
        self.llama_model_path = os.environ.get('LLAMA2_7B_ORIGIN_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2   
            
    def completion_to_prompt(completion):
        return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n" 
    
    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"
        return prompt
    
    def test_bigdl_llm(self):    
        llm = IpexLLM.from_model_id(
            model_name=self.llama_model_path,
            tokenizer_name=self.llama_model_path,
            context_window=512,
            max_new_tokens=32,
            model_kwargs={},
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            device_map="xpu",
        )
        res = llm.complete("What is AI?")
        assert res!=None
        

if __name__ == '__main__':
    pytest.main([__file__])