from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM
from vllm import SamplingParams
from transformers import AutoTokenizer
import requests


model_path = "/llm/models/MiniCPM-V-2_6"
model_path = "/llm/models/Qwen2-VL-7B-Instruct"
prompt = "What is in the image?"

def run_minicpmv(question, modality):
    assert modality == "image"
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    # 2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt, stop_token_ids

def run_qwen2_vl(question, modality):
    assert modality == "image"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return prompt, stop_token_ids

model_example_map = {
    "minicpmv": run_minicpmv,
    "qwen2_vl": run_qwen2_vl,
}

llm = LLM(
          model=model_path,
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="fp8",
          tensor_parallel_size=1,
          disable_async_output_proc=True,
          distributed_executor_backend="ray",
          max_model_len=4000,
          trust_remote_code=True,
          block_size=8,
          max_num_batched_tokens=4000)


model_type = llm.llm_engine.model_config.hf_config.model_type
prompt, stop_token_ids = model_example_map[model_type](prompt, "image")


# Load the image using PIL.Image
from PIL import Image
image_url="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')


sampling_params = SamplingParams(temperature=0.1,
                                 top_p=0.001,
                                 repetition_penalty=1.05,
                                 max_tokens=64,
                                 stop_token_ids=stop_token_ids)


# Single prompt inference
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
}, sampling_params=sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)


