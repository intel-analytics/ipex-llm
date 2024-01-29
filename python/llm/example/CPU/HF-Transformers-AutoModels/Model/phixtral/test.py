import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
model_name = "phixtral-4x2_8"
instruction = "What is AI"

torch.set_default_device("cpu")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/disk1/models/phixtral-4x2_8", 
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/disk1/models/phixtral-4x2_8", 
    trust_remote_code=True
)

# Tokenize the input string
inputs = tokenizer(
    instruction, 
    return_tensors="pt", 
    return_attention_mask=False
)

# Generate text using the model
start_time = time.time()
outputs = model.generate(**inputs, max_length=128)
end_time = time.time()

# Decode and print the output
text = tokenizer.batch_decode(outputs)[0]
print(text)
print(end_time - start_time)
