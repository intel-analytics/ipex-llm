import torch, transformers
import intel_extension_for_pytorch as ipex
import sys, os, time
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
# from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm import optimize_model

print("Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/disk1/models/phixtral-4x2_8",
    trust_remote_code=True
)

print("Creating model...")
# model = AutoModelForCausalLM.from_pretrained('/mnt/disk1/models/Yuan2-2B-hf', trust_remote_code=True, load_in_4bit=True, optimize_model=False).eval()
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/disk1/models/phixtral-4x2_8",
    torch_dtype=torch.float16, 
    trust_remote_code=True).eval()
model = optimize_model(model)
model = model.to('xpu')

inputs = tokenizer("请问目前最先进的机器学习算法有哪些？", return_tensors="pt")["input_ids"]

# Warmup
model.generate(inputs.to('xpu'), do_sample=True, top_k=5, max_length=128)
print('Finish warmup')

# 运行generate三次并计算平均时间
total_time = 0
for _ in range(3):
    start_time = time.time()
    outputs = model.generate(inputs.to('xpu'), do_sample=True, top_k=5, max_length=128)
    end_time = time.time()
    total_time += end_time - start_time

# 计算平均时间
average_time = total_time / 3
print("Average generation time: {:.2f} seconds".format(average_time))
print(tokenizer.decode(outputs[0]))