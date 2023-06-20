import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

if __name__ == '__main__':
    model_path = 'decapoda-research/llama-7b-hf'
    model = LlamaForCausalLM.from_pretrained(
    model_path,
    device_map='cpu',
    torch_dtype=torch.float32,
    )

    from bigdl.llm.transformers import quantize_4bit

    model = quantize_4bit(model)
    print(model)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    input_str = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

    with torch.inference_mode():
        input_ids = tokenizer.encode(input_str, return_tensors="pt")
        output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)
