from transformers import AutoModelForCausalLM

model_path = "facebook/opt-30b"
model = AutoModelForCausalLM.from_pretrained(model_path)