import os
import pytest
import torch
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH'))
    ])
def test_optimize_model(Model, Tokenizer, model_path):
    with torch.inference_mode():
        tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        model = Model.from_pretrained(model_path,
                                    load_in_4bit=True,
                                    optimize_model=False,
                                    trust_remote_code=True)
        model = model.to(device)
        logits_base_model = (model(input_ids)).logits
        model.to('cpu')  # deallocate gpu memory

        model = Model.from_pretrained(model_path,
                                    load_in_4bit=True,
                                    optimize_model=True,
                                    trust_remote_code=True)
        model = model.to(device)
        logits_optimized_model = (model(input_ids)).logits
        model.to('cpu')

        tol = 1e-02
        num_false = torch.isclose(logits_optimized_model, logits_base_model, rtol=tol, atol=tol)\
            .flatten().tolist().count(False)
        percent_false = num_false / logits_optimized_model.numel()
        assert percent_false < 1e-02
    