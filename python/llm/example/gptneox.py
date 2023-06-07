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


# In this example, we show a pipeline to conduct inference on a converted
# low-precision (int4) large language model in gptneox family,
# using bigdl-llm

# Step 1: (convert) and load the low-precision model
from bigdl.llm.ggml.transformers import AutoModelForCausalLM

# here you may input the HuggingFace repo id directly as the value of `pretrained_model_name_or_path`.
# This will allow the pre-trained model to be downloaded directly from the HuggingFace repository.
# The downloaded model will then be converted to binary format with int4 dtype weights,
# and saved into the cache_dir folder.
#
# if you already have the pre-trained model downloaded, you can provide the path to
# the downloaded folder as the value of `pretrained_model_name_or_path``
llm = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path='togethercomputer/RedPajama-INCITE-7B-Chat',
    model_family='gptneox',
    cache_dir='./',
    n_threads=28)

# if you want to explicitly convert the pre-trained model,
# you can use the `convert_model` API to convert it first,
# and then load the binary checkpoint directly.
#
# from bigdl.llm.ggml import convert_model
#
# output_ckpt_path = convert_model(
#     input_path="path/to/huggingface/checkpoint/folder",
#     output_path="folder/where/int4/converted-model/saved",
#     model_family='gptneox')
#
# llm = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=output_ckpt_path,
#     model_family='gptneox',
#     n_threads=28)

# Step 2: Conduct infernece
import time

# Option 1: Use transformers tokenizer
from transformers import AutoTokenizer

# here you should use the same repo id
tokenizer = AutoTokenizer.from_pretrained('togethercomputer/RedPajama-INCITE-7B-Chat')

st = time.time()

tokens_id = tokenizer.encode("Q: Tell me something about Intel. A:")
output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
output = tokenizer.decode(output_tokens_id)

print(f"Inference time (transformers tokenizer): {time.time()-st} s")
print(f"Output:\n{output}")

# Option 2: Use bigdl-llm based tokenizer
st = time.time()

tokens_id = llm.tokenize("Q: Tell me something about Intel. A:")
output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
output = llm.decode(output_tokens_id)

print(f"Inference time (bigdl-llm based tokenizer): {time.time()-st} s")
print(f"Output:\n{output}")

# Option 3: fast forward
st = time.time()

output = llm(prompt="Q: Tell me something about Intel. A:", max_tokens=32)

print(f"Inference time (fast forward): {time.time()-st} s")
print(f"Output:\n{output}")
