import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import intel_extension_for_pytorch as ipex


tokenizer = AutoTokenizer.from_pretrained("/mnt/disk1/models/internlm2-chat-7b/", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/mnt/disk1/models/internlm2-chat-7b/", torch_dtype=torch.float16, trust_remote_code=True)
model = model.to("xpu")
model = model.eval()
start_time = time.time()
response, history = model.chat(tokenizer, "解释一种机器学习算法", history=[],max_new_tokens=128)
end_time = time.time()
print(response)
print(end_time-start_time)
# Hello! How can I help you today?
# response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
# print(response)
