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

import os
import pytest
import tempfile

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


mistral_model_path = os.environ.get('MISTRAL_ORIGIN_PATH')

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

@pytest.mark.parametrize("Model, Tokenizer, model_path, prompt", [
    (AutoModelForCausalLM, AutoTokenizer, mistral_model_path, prompt)
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

@pytest.mark.parametrize('prompt, answer', [
    ('What is the capital of France?\n\n', 'Paris')
    ])
@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, AutoTokenizer, mistral_model_path),
    ])
def test_load_low_bit_completion(Model, Tokenizer, model_path, prompt, answer):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Model.from_pretrained(model_path,
                                  load_in_4bit=True,
                                  optimize_model=True,
                                  trust_remote_code=True)
    
    with tempfile.TemporaryDirectory() as tempdir:
        model.save_low_bit(tempdir)
        loaded_model = Model.load_low_bit(tempdir,
                                          optimize_model=True,
                                          trust_remote_code=True)

        with torch.inference_mode():
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output = loaded_model.generate(input_ids, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)

            assert answer in output_str

if __name__ == '__main__':
    pytest.main([__file__])
