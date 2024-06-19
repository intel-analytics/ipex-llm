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
# Some parts of this file is adapted from
# https://github.com/haotian-liu/LLaVA/blob/v1.1.1/llava/model/builder.py
# and
# https://github.com/haotian-liu/LLaVA/blob/v1.1.1/llava/serve/cli.py
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import torch
import time

from transformers import AutoModelForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)

from ipex_llm import optimize_model

# Load the pretrained model.
# Adapted from llava.model.builder.load_pretrained_model.
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cpu"):
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float32

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided.'
                          'If you are loading a LoRA model, please provide the `model_base` argument'
                          '. Detailed instruction:'
                          'https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(
                    token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(
                    token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path,
                                                              'non_lora_trainables.bin'),
                                                 map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(
                    model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith(
                'base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith(
                    'model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(
                        model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(
                model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float32)
                                    for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP32...')
            model.to(torch.float32)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(
            model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(
            model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float32)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

# Initialize conversation from templates and get conversation roles.
def get_conv_and_role(model_name):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    return conv, roles

# Load image from a url or path.
def load_image(image_file):
    import requests
    from PIL import Image
    from io import BytesIO

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def generate_image_tensor(image_file):
    image = load_image(image_file)
    model_cfg = {"image_aspect_ratio": 'pad'}
    image_tensor = process_images([image], image_processor, model_cfg)
    return image_tensor

# Generate input prompt with user input.
def get_prompt(mm_use_im_start_end, first_round, conv, user_input):
    if first_round:
        # first message
        if mm_use_im_start_end:
            user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + user_input
        else:
            user_input = DEFAULT_IMAGE_TOKEN + '\n' + user_input
        conv.append_message(conv.roles[0], user_input)
    else:
        # later messages
        conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def get_stopping_criteria(conv, tokenizer, input_ids):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    return stopping_criteria


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict Tokens using `generate()` API for LLaVA model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="liuhaotian/llava-v1.5-13b",
                        help='The huggingface repo id for the LLaVA model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-path-or-url', type=str,
                        required=True, help='Image path or url for the input image that the chat will focus on')
    parser.add_argument('--n-predict', type=int, default=512,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    model_name = get_model_name_from_path(model_path)

    # Disable the redundant torch default initialization to accelerate model creation.
    disable_torch_init()

    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path=model_path,
                                                                 model_base=None,
                                                                 model_name=model_name,
                                                                 device_map=None)

    # With only one line to enable IPEX-LLM optimization on model
    model = optimize_model(model)

    # Generate image tensor
    image_tensor = generate_image_tensor(args.image_path_or_url)

    # Get conversation template and roles
    conv, roles = get_conv_and_role(model_name)

    first_round = True
    while True:
        try:
            user_input = input(f"{roles[0]}: ")
        except EOFError:
            user_input = ""
        if not user_input:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        prompt = get_prompt(model.config.mm_use_im_start_end, first_round, conv, user_input)
        first_round = False
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        stopping_criteria = get_stopping_criteria(conv, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate predicted tokens
        with torch.inference_mode():
            st = time.time()
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                max_new_tokens=args.n_predict,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            end = time.time()
            #print(f'Inference time: {end-st} s')

        outputs = tokenizer.decode(output_ids[0, :], skip_special_tokens=True).strip()
        conv.messages[-1][-1] = outputs
