import torch
import os
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import time
import numpy as np
from benchmark_util import BenchmarkWrapper

model_path ='/opt/occlum_spark/data/models/vicuna-7b-bigdl/'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
model = BenchmarkWrapper(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

with torch.inference_mode():
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print("input length is : ", input_ids.shape[1])
    output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
