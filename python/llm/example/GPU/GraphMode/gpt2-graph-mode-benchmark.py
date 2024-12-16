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
# This is modified from https://github.com/intel-sandbox/customer-ai-test-code/blob/main/gpt2-benchmark-for-sangfor.py
#
import torch
import time
import argparse
import contextlib
from transformers import GPT2ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Qwen2ForSequenceClassification
from torch.profiler import profile, record_function, ProfilerActivity, schedule


# Get the batch size and device
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=1, help='an integer for the batch size')
parser.add_argument('--device', type=str, default='cpu', help='an string for the device')
parser.add_argument('--profile', type=bool, default=False, help='enable protch profiler for CPU/XPU')
parser.add_argument('--engine', type=str, default='ipex-llm', help='an string for the device')
parser.add_argument('--model_path', type=str, help='an string for the device')
args = parser.parse_args()
enable_profile=args.profile
batch_size = args.batch_size
device = args.device
engine = args.engine
model_path = args.model_path
print(f"The batch size is: {batch_size}, device is {device}")

def profiler_setup(profiling=False, *args, **kwargs):
    if profiling:
        return torch.profiler.profile(*args, **kwargs)
    else:
        return contextlib.nullcontext()

my_schedule = schedule(
    skip_first=6,
    wait=1,
    warmup=1,
    active=1
    )

# define a handler for outputing results
def trace_handler(p):
    if(device == 'xpu'):
        print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=20))
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=20))

dtype = torch.bfloat16 if device == 'cpu' else torch.float16
num_labels = 5
model_name = model_path
model_name = model_name + "-classification"
model_name_ov = model_name + "-ov"
model_name_ov = model_name_ov + "-fp16"

if (engine == 'ipex') :
    import torch
    import intel_extension_for_pytorch as ipex
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype,
                                                               pad_token_id=tokenizer.eos_token_id,
                                                               low_cpu_mem_usage=True
                                                               ).eval().to(device)
elif (engine == 'ipex-llm'):
    from ipex_llm.transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               torch_dtype=dtype,
                                                               load_in_low_bit="fp16",
                                                               pad_token_id=tokenizer.eos_token_id,
                                                               low_cpu_mem_usage=True).to(device)
    model = torch.compile(model, backend='inductor')
    print(model)
else:
    from optimum.intel import OVModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name_ov, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = OVModelForSequenceClassification.from_pretrained(model_name_ov, torch_dtype=dtype).to(device)

# Intel(R) Extension for PyTorch*
if engine == 'ipex':
    if device == 'cpu':
        model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)
        model = torch.compile(model, backend='ipex')
    else:    # Intel XPU
        model = ipex.optimize(model, dtype=dtype, inplace=True)
    model=torch.compile(model, backend="inductor")
    print(model)
elif engine == 'ov':
    print("OV inference")


prompt = ["this is the first prompt"]
prompts = prompt * batch_size
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)

if engine == 'ipex' or engine == 'ipex-llm':
    inputs.to(device)
    elapsed_times = []
    with profiler_setup(profiling=enable_profile, activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        schedule=my_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True
        ) as prof:
        for i in range(10):
            start_time = time.time()
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
            predicted_class_ids = logits.argmax(dim=1).tolist()
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            if(enable_profile):
                prof.step()
elif engine == 'ov':
    print("OV inference")
    elapsed_times = []
    for i in range(10):
        start_time = time.time()
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_ids = logits.argmax(dim=1).tolist()
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

# Skip the first two values and calculate the average of the remaining elapsed times
average_elapsed_time = sum(elapsed_times[2:]) / len(elapsed_times[2:])
classfication_per_second = batch_size/average_elapsed_time
print(f"Average time taken (excluding the first two loops): {average_elapsed_time:.4f} seconds, Classification per seconds is {classfication_per_second:.4f}")
