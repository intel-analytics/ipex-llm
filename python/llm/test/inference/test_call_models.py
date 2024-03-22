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


from ipex_llm.models import Llama, Bloom, Gptneox, Starcoder
from ipex_llm.transformers import LlamaForCausalLM, BloomForCausalLM, \
    GptneoxForCausalLM, StarcoderForCausalLM
import pytest
from unittest import TestCase
import os


class Test_Models_Basics(TestCase):

    def setUp(self):
        self.llama_model_path = os.environ.get('LLAMA_INT4_CKPT_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_INT4_CKPT_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_INT4_CKPT_PATH')
        self.starcoder_model_path = os.environ.get('STARCODER_INT4_CKPT_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2         

    def test_llama_completion_success(self):
        llm = Llama(self.llama_model_path, n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=False)
        # assert "Paris" in output['choices'][0]['text']

    def test_llama_completion_with_stream_success(self):
        llm = Llama(self.llama_model_path, n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=True)

    def test_llama_for_causallm(self):
        llm = LlamaForCausalLM.from_pretrained(self.llama_model_path, native=True,
                                               n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=False)

    def test_bloom_completion_success(self):
        llm = Bloom(self.bloom_model_path, n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=False)
        # avx = get_avx_flags()
        # if avx == "_avx512":
        #     # For platforms without avx512, the current text completion may output gibberish
        #     assert "Paris" in output['choices'][0]['text']

    def test_bloom_completion_with_stream_success(self):
        llm = Bloom(self.bloom_model_path, n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=True)

    def test_bloom_for_causallm(self):
        llm = BloomForCausalLM.from_pretrained(self.bloom_model_path, native=True,
                                               n_threads=self.n_threads)
        output = llm("What is the capital of France?", max_tokens=32, stream=False)

    def test_gptneox_completion_success(self):
        llm = Gptneox(self.gptneox_model_path, n_threads=self.n_threads)
        output = llm("Q: What is the capital of France? A:", max_tokens=32, stream=False)
        # assert "Paris" in output['choices'][0]['text']

    def test_gptneox_completion_with_stream_success(self):
        llm = Gptneox(self.gptneox_model_path, n_threads=self.n_threads)
        output = llm("Q: What is the capital of France? A:", max_tokens=32, stream=True)

    def test_getneox_for_causallm(self):
        llm = GptneoxForCausalLM.from_pretrained(self.gptneox_model_path, native=True,
                                                 n_threads=self.n_threads)
        output = llm("Q: What is the capital of France? A:", max_tokens=32, stream=False)
    
    def test_starcoder_completion_success(self):
        llm = Starcoder(self.starcoder_model_path, n_threads=self.n_threads)
        output = llm("def print_hello_world(", max_tokens=32, stream=False)
        # assert "Paris" in output['choices'][0]['text']

    def test_starcoder_completion_with_stream_success(self):
        llm = Starcoder(self.starcoder_model_path, n_threads=self.n_threads)
        output = llm("def print_hello_world(", max_tokens=32, stream=True)

    def test_starcoder_for_causallm(self):
        llm = StarcoderForCausalLM.from_pretrained(self.starcoder_model_path, native=True,
                                                   n_threads=self.n_threads)
        output = llm("def print_hello_world(", max_tokens=32, stream=False)


if __name__ == '__main__':
    pytest.main([__file__])
