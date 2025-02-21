#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import torch
import argparse
from ipex_llm.transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images
 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Predict Tokens using generate() API for Janus-Pro model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="deepseek-ai/Janus-Pro-7B",
                        help='The Hugging Face repo id for the Janus-Pro model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--image-path', type=str,
                        help='The path to the image for inference.')
    parser.add_argument('--prompt', type=str,
                        help='Prompt for inference.')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    
    args = parser.parse_args()

    model_path = args.repo_id_or_model_path
    model_name = os.path.basename(model_path)
    prompt = args.prompt
    image_path = args.image_path
    if prompt is None:
        if image_path is not None and os.path.exists(image_path):
            prompt = "Describe the image in detail."
        else:
            prompt = "What is AI?"
    
    # The following code is adapted from 
    # https://github.com/deepseek-ai/Janus?tab=readme-ov-file#multimodal-understanding
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    model_vl = AutoModelForCausalLM.from_pretrained(
        model_path, 
        load_in_low_bit=args.low_bit,
        optimize_model=True,
        trust_remote_code=True,
        modules_to_not_convert=["vision_model"]
    ).eval()
    
    model_vl = model_vl.half().to('xpu')
    
    if image_path is not None and os.path.exists(image_path):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{prompt}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    )
    
    prepare_inputs = prepare_inputs.to(device='xpu', dtype=torch.half)
    
    # run image encoder to get the image embeddings
    inputs_embeds = model_vl.prepare_inputs_embeds(**prepare_inputs)
    
    with torch.inference_mode():
        # ipex_llm model needs a warmup, then inference time can be accurate
        outputs = model_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.n_predict,
            do_sample=False,
            use_cache=True,
        )

        st = time.time()
        # run the model to get the response
        outputs = model_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.n_predict,
            do_sample=False,
            use_cache=True,
        )
        ed = time.time()
        
        reponse = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    print(f'Inference time: {ed-st} s')
    print('-'*20, 'Input Image Path', '-'*20)
    print(image_path)
    print('-'*20, 'Input Prompt (Formatted)', '-'*20)
    print(f"{prepare_inputs['sft_format'][0]}")
    print('-'*20, 'Chat Output', '-'*20)
    print(reponse)
