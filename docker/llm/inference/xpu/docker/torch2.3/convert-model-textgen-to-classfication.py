import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM


dtype=torch.bfloat16
num_labels = 5

model_name="/llm/models/gpt2-medium"
# model_name="/home/llm/local_models/Qwen/Qwen2-0.5B-Instruct"
# model_name="/home/llm/disk/llm/EleutherAI/gpt-neo-1.3B"
# model_name="/home/llm/disk/llm/facebook/opt-350m"

save_directory = model_name + "-classification"

# Initialize the tokenizer
# Need padding from the left and padding to 1024
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_directory)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, pad_token_id=tokenizer.eos_token_id,)
config = AutoConfig.from_pretrained(model_name)
print("text gen model")
print(model)
print(config)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=dtype)
save_directory = model_name + "-classification"
model.save_pretrained(save_directory)    


model = AutoModelForSequenceClassification.from_pretrained(save_directory, torch_dtype=dtype, pad_token_id=tokenizer.eos_token_id)
config = AutoConfig.from_pretrained(save_directory)
print("text classification model")
print(model)
print(config)
