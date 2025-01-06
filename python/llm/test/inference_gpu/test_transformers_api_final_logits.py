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
import gc
import pytest

import torch
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

device = os.environ['DEVICE']
print(f'Running on {device}')

PROMPT = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
TEST_MODEL_LIST = [
    ("MPT-7B", AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
]


@pytest.mark.parametrize('Name, Model, Tokenizer, model_path',TEST_MODEL_LIST)
def test_optimize_model(Name, Model, Tokenizer, model_path):
    with torch.inference_mode():
        tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

        model = Model.from_pretrained(model_path,
                                    load_in_4bit=True,
                                    optimize_model=False,
                                    trust_remote_code=True)
        model = model.to(device)

        logits_base_model = (model(input_ids)).logits

        model.to('cpu')  # deallocate gpu memory

        opt_model = Model.from_pretrained(model_path,
                                    load_in_4bit=True,
                                    optimize_model=True,
                                    trust_remote_code=True)
        opt_model = opt_model.to(device)
        logits_optimized_model = (opt_model(input_ids)).logits
        opt_model.to('cpu')

        tol = 1e-03
        num_false = torch.isclose(logits_optimized_model, logits_base_model, rtol=tol, atol=tol)\
            .flatten().tolist().count(False)

        percent_false = num_false / logits_optimized_model.numel()
        torch.xpu.empty_cache()
        del model
        del opt_model
        gc.collect()
        assert percent_false < 1e-02


if __name__ == '__main__':
    pytest.main([__file__])
