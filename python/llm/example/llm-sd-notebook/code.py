# the code copy of the llm-sd-notebook demo


import os
import time
import torch
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer

model_path = os.environ.get('LLM_MODEL_PATH')
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

input_str = "Give me a text prompt to generate an image of a castle\n\n"
with torch.inference_mode():
    st = time.time()
    input_ids = tokenizer.encode(input_str, return_tensors="pt")
    output = model.generate(input_ids, do_sample=False, max_new_tokens=64)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    end = time.time()
print('Prompt:', input_str)
print('Output:', output_str)
print(f'Inference time: {end-st} s')

prompt = output_str.split('"')[-1]
from bigdl_diffusion.diffusers.pipelines import *
pipe = NanoStableDiffusionPipeline.from_pretrained(os.getenv('SD_MODEL_PATH'), device='GPU', backend='ov')
output = pipe(prompt, num_inference_steps=10).images[0]
output.show()





