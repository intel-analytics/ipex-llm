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

from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

device = os.environ['DEVICE']
print(f'Running on {device}')
if device == 'xpu':
    import intel_extension_for_pytorch as ipex

@pytest.mark.parametrize('prompt, answer', [
    ('What is the capital of France?\n\n', 'Paris')
    ])
@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    (AutoModel, AutoTokenizer, os.environ.get('CHATGLM2_6B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('FALCON_7B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    ])
def test_completion(Model, Tokenizer, model_path, prompt, answer):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Model.from_pretrained(model_path,
                                load_in_4bit=True,
                                optimize_model=True,
                                trust_remote_code=True)
    model = model.to(device)  # deallocate gpu memory

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=32)
    model.to('cpu')
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)

    assert answer in output_str
        
if __name__ == '__main__':
    pytest.main([__file__])
