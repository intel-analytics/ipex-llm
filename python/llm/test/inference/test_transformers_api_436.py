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


import unittest
import os
import tempfile
import time
import torch
import pytest

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer


prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

@pytest.mark.parametrize("Model, Tokenizer, model_path, prompt", [
    (AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA_ORIGIN_PATH'), prompt),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MIXTRAL_ORIGIN_PATH'), prompt),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('FALCON_7B_ORIGIN_PATH'), prompt)
])
    
def test_optimize_model(Model, Tokenizer, model_path, prompt):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=False,
                                trust_remote_code=True)
    logits_base_model = (model(input_ids)).logits

    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=True,
                                trust_remote_code=True)
    logits_optimized_model = (model(input_ids)).logits
    diff = abs(logits_base_model - logits_optimized_model).flatten()

    assert any(diff) is False


if __name__ == '__main__':
    pytest.main([__file__])
