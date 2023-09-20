import os

import torch
import transformers
from transformers import LlamaTokenizer  # noqa: F402
from bigdl.llm.transformers.qlora import PeftModel
from bigdl.llm.transformers import AutoModelForCausalLM
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--adapter_path', type=str,)
    parser.add_argument('--output_path', type=str,)

    args = parser.parse_args()
    base_model = model_path = args.repo_id_or_model_path
    adapter_path = args.adapter_path
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_low_bit="nf4", # should load the orignal model
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    base_model.save_pretrained(args.output_path, state_dict=deloreanized_sd)
    tokenizer.save_pretrained(args.output_path)
