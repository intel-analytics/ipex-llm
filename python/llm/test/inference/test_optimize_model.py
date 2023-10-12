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

import pytest
import os
from unittest import TestCase

class TestOptimizeModel(TestCase):

    def setUp(self):        
        self.llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_ORIGIN_PATH')
        self.starcoder_model_path = os.environ.get('STARCODER_ORIGIN_PATH')

        self.prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

    def test_optimize_llama(self):
        from bigdl.llm.transformers import AutoModelForCausalLM
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(self.llama_model_path)
        input_ids = tokenizer.encode(self.prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(self.llama_model_path,
                                             load_in_4bit=True,
                                             optimize_model=False)
        logits_base_model = (model(input_ids)).logits

        model = AutoModelForCausalLM.from_pretrained(self.llama_model_path,
                                             load_in_4bit=True,
                                             optimize_model=True)
        logits_optimized_model = (model(input_ids)).logits
        diff = abs(logits_base_model - logits_optimized_model).flatten()

        assert any(diff) is False

    def test_optimize_bloom(self):
        from bigdl.llm.transformers import AutoModelForCausalLM
        from transformers import BloomTokenizerFast

        tokenizer = BloomTokenizerFast.from_pretrained(self.bloom_model_path)
        input_ids = tokenizer.encode(self.prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(self.bloom_model_path,
                                             load_in_4bit=True,
                                             optimize_model=False)
        logits_base_model = (model(input_ids)).logits

        model = AutoModelForCausalLM.from_pretrained(self.bloom_model_path,
                                             load_in_4bit=True,
                                             optimize_model=True)
        logits_optimized_model = (model(input_ids)).logits
        diff = abs(logits_base_model - logits_optimized_model).flatten()

        assert any(diff) is False

    def test_optimize_gptneox(self):
        from bigdl.llm.transformers import AutoModelForCausalLM
        from transformers import GPTNeoXTokenizerFast

        tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.gptneox_model_path)
        input_ids = tokenizer.encode(self.prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(self.gptneox_model_path,
                                             load_in_4bit=True,
                                             optimize_model=False)
        logits_base_model = (model(input_ids)).logits

        model = AutoModelForCausalLM.from_pretrained(self.gptneox_model_path,
                                             load_in_4bit=True,
                                             optimize_model=True)
        logits_optimized_model = (model(input_ids)).logits
        diff = abs(logits_base_model - logits_optimized_model).flatten()

        assert any(diff) is False

    def test_optimize_starcoder(self):
        from bigdl.llm.transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.starcoder_model_path)
        input_ids = tokenizer.encode(self.prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(self.starcoder_model_path,
                                             load_in_4bit=True,
                                             optimize_model=False)
        logits_base_model = (model(input_ids)).logits

        model = AutoModelForCausalLM.from_pretrained(self.starcoder_model_path,
                                             load_in_4bit=True,
                                             optimize_model=True)
        logits_optimized_model = (model(input_ids)).logits
        diff = abs(logits_base_model - logits_optimized_model).flatten()

        assert any(diff) is False


if __name__ == '__main__':
    pytest.main([__file__])
