import torch

lora_path = "/home/wangruonan/yina/BigDL/python/llm/example/GPU/QLoRA-FineTuning/alpaca-qlora/bigdl-qalora-alpaca-2/checkpoint-1100/adapter_model.bin"
lora = torch.load(lora_path, map_location='cpu')
tmp_keys = [key[17:-14] for key in lora.keys() if 'lora_A' in key]

for tmp_key in tmp_keys:
    a = lora['base_model.model.'+tmp_key+'.lora_B.weight']
    b = lora['base_model.model.'+tmp_key+'.lora_A.weight']
    lora['base_model.model.'+tmp_key+'.lora_B.weight'] = torch.repeat_interleave(a, 64, dim=1) / 4 / 64
    lora['base_model.model.'+tmp_key+'.lora_B.weight'] = torch.repeat_interleave(b, 64, dim=1) / 4 / 64

# TODO: peft merge
