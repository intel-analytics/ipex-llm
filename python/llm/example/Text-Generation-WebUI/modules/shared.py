import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

from modules.logging_colors import logger

# Model variables
model = None
tokenizer = None
model_name = 'None'
is_seq2seq = False
model_dirty_from_training = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*Is typing...*'

# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False

# UI defaults
settings = {
    'dark_theme': True,
    'show_controls': True,
    'start_with': '',
    'mode': 'chat',
    'chat_style': 'cai-chat',
    'prompt-default': 'QA',
    'prompt-notebook': 'QA',
    'preset': 'simple-1',
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'negative_prompt': '',
    'seed': -1,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 200000,
    'max_tokens_second': 0,
    'max_updates_second': 0,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'auto_max_new_tokens': False,
    'ban_eos_token': False,
    'add_bos_token': True,
    'skip_special_tokens': True,
    'stream': True,
    'character': 'Assistant',
    'name1': 'You',
    'custom_system_message': '',
    'instruction_template_str': "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    'chat_template_str': "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}",
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'autoload_model': False,
    'gallery-items_per_page': 50,
    'gallery-open': False,
    'default_extensions': ['gallery'],
}


# Parser copied from https://github.com/vladmandic/automatic
parser = argparse.ArgumentParser(description="Text generation web UI", conflict_handler='resolve', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))

# Basic settings
group = parser.add_argument_group('Basic settings')
group.add_argument('--multi-user', action='store_true', help='Multi-user mode. Chat histories are not saved or automatically loaded. Warning: this is likely not safe for sharing publicly.')
group.add_argument('--character', type=str, help='The name of the character to load in chat mode by default.')
group.add_argument('--model', type=str, help='Name of the model to load by default.')
group.add_argument('--lora', type=str, nargs='+', help='The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.')
group.add_argument('--model-dir', type=str, default='models/', help='Path to directory with all the models.')
group.add_argument('--lora-dir', type=str, default='loras/', help='Path to directory with all the loras.')
group.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
group.add_argument('--settings', type=str, help='Load the default interface settings from this yaml file. See settings-template.yaml for an example. If you create a file called settings.yaml, this file will be loaded by default without the need to use the --settings flag.')
group.add_argument('--extensions', type=str, nargs='+', help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
group.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
group.add_argument('--chat-buttons', action='store_true', help='Show buttons on the chat tab instead of a hover menu.')

# Model loader
group = parser.add_argument_group('Model loader')
group.add_argument('--loader', type=str, help='Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2, AutoGPTQ, AutoAWQ, GPTQ-for-LLaMa, ctransformers, QuIP#.')

# Transformers/Accelerate
group = parser.add_argument_group('Transformers/Accelerate')
group.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
group.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
group.add_argument('--gpu-memory', type=str, nargs='+', help='Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
group.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
group.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
group.add_argument('--disk-cache-dir', type=str, default='cache', help='Directory to save the disk cache to. Defaults to "cache".')
group.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision (using bitsandbytes).')
group.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
group.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.')
group.add_argument('--trust-remote-code', action='store_true', help='Set trust_remote_code=True while loading the model. Necessary for some models.')
group.add_argument('--force-safetensors', action='store_true', help='Set use_safetensors=True while loading the model. This prevents arbitrary code execution.')
group.add_argument('--no_use_fast', action='store_true', help='Set use_fast=False while loading the tokenizer (it\'s True by default). Use this if you have any problems related to use_fast.')
group.add_argument('--use_flash_attention_2', action='store_true', help='Set use_flash_attention_2=True while loading the model.')

# bitsandbytes 4-bit
group = parser.add_argument_group('bitsandbytes 4-bit')
group.add_argument('--load-in-4bit', action='store_true', help='Load the model with 4-bit precision (using bitsandbytes).')
group.add_argument('--use_double_quant', action='store_true', help='use_double_quant for 4-bit.')
group.add_argument('--compute_dtype', type=str, default='float16', help='compute dtype for 4-bit. Valid options: bfloat16, float16, float32.')
group.add_argument('--quant_type', type=str, default='nf4', help='quant_type for 4-bit. Valid options: nf4, fp4.')

# BigDL-LLM
group = parser.add_argument_group('BigDL-LLM')
group.add_argument('--device', type=str, default='cpu', help='the device type, it could be CPU or GPU')
group.add_argument('--load-in-4bit', action='store_true', default=False, help='boolean value, True means loading linear’s weight to symmetric int 4 if'\
                   'the model is a regular fp16/bf16/fp32 model, and to asymmetric int 4 if the model is GPTQ model.Default to be False')
group.add_argument('--load-in-low-bit', type=str, default=None, help='str value, options are sym_int4, asym_int4, sym_int5, asym_int5'\
                   ', sym_int8, nf3, nf4, fp4, fp8, fp8_e4m3, fp8_e5m2, fp16 or bf16. sym_int4 means symmetric int 4, asym_int4 means asymmetric int 4,'\
                   'nf4 means 4-bit NormalFloat, etc. Relevant low bit optimizations will be applied to the model.')
group.add_argument('--optimize-model', action='store_true', help='boolean value, Whether to further optimize the low_bit llm model.')
group.add_argument('--modules-to-not-convert', type=str, default=None, help='list of str value, modules (nn.Module) that are skipped when conducting model optimizations.')
group.add_argument('--cpu-embedding', action='store_true', help='Whether to replace the Embedding layer, may need to set it to `True` when running BigDL-LLM on GPU on Windows. Default to be `False`')
group.add_argument('--lightweight-bmm', action='store_true', help='Whether to replace the torch.bmm ops, may need to set it to `True` when running BigDL-LLM on GPU on Windows.')
group.add_argument('--use-cache', action='store_true', help='If use_cache is True, past key values are used to speed up decoding if applicable to model.')
group.add_argument('--trust-remote-code', action='store_true', help='Set trust_remote_code=True while loading the model. Necessary for some models.')

# Gradio
group = parser.add_argument_group('Gradio')
group.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
group.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
group.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
group.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
group.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
group.add_argument('--gradio-auth', type=str, help='Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".', default=None)
group.add_argument('--gradio-auth-path', type=str, help='Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.', default=None)
group.add_argument('--ssl-keyfile', type=str, help='The path to the SSL certificate key file.', default=None)
group.add_argument('--ssl-certfile', type=str, help='The path to the SSL certificate cert file.', default=None)

# API
group = parser.add_argument_group('API')
group.add_argument('--api', action='store_true', help='Enable the API extension.')
group.add_argument('--public-api', action='store_true', help='Create a public URL for the API using Cloudfare.')
group.add_argument('--public-api-id', type=str, help='Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.', default=None)
group.add_argument('--api-port', type=int, default=5000, help='The listening port for the API.')
group.add_argument('--api-key', type=str, default='', help='API authentication key.')
group.add_argument('--admin-key', type=str, default='', help='API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.')
group.add_argument('--nowebui', action='store_true', help='Do not launch the Gradio UI. Useful for launching the API in standalone mode.')

# Multimodal
group = parser.add_argument_group('Multimodal')
group.add_argument('--multimodal-pipeline', type=str, default=None, help='The multimodal pipeline to use. Examples: llava-7b, llava-13b.')

# Deprecated parameters
# group = parser.add_argument_group('Deprecated')

args = parser.parse_args()
args_defaults = parser.parse_args([])
provided_arguments = []
for arg in sys.argv[1:]:
    arg = arg.lstrip('-').replace('-', '_')
    if hasattr(args, arg):
        provided_arguments.append(arg)

deprecated_args = []


def do_cmd_flags_warnings():

    # Deprecation warnings
    for k in deprecated_args:
        if getattr(args, k):
            logger.warning(f'The --{k} flag has been deprecated and will be removed soon. Please remove that flag.')

    # Security warnings
    if args.trust_remote_code:
        logger.warning('trust_remote_code is enabled. This is dangerous.')
    if 'COLAB_GPU' not in os.environ and not args.nowebui:
        if args.share:
            logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")
        if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
            logger.warning("\nYou are potentially exposing the web UI to the entire internet without any access password.\nYou can create one with the \"--gradio-auth\" flag like this:\n\n--gradio-auth username:password\n\nMake sure to replace username:password with your own.")
            if args.multi_user:
                logger.warning('\nThe multi-user mode is highly experimental and should not be shared publicly.')


def fix_loader_name(name):
    if not name:
        return name

    name = name.lower()
    if name in ['transformers', 'huggingface', 'hf', 'hugging_face', 'hugging face']:
        return 'Transformers'
    elif name in ['BigDL-LLM', 'bigdl-llm']:
        return 'BigDL-LLM'


def add_extension(name, last=False):
    if args.extensions is None:
        args.extensions = [name]
    elif last:
        args.extensions = [x for x in args.extensions if x != name]
        args.extensions.append(name)
    elif name not in args.extensions:
        args.extensions.append(name)


def is_chat():
    return True


args.loader = fix_loader_name(args.loader)

# Activate the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')

# Activate the API extension
if args.api or args.public_api:
    add_extension('openai', last=True)

# Load model-specific settings
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Load custom model-specific settings
with Path(f'{args.model_dir}/config-user.yaml') as p:
    if p.exists():
        user_config = yaml.safe_load(open(p, 'r').read())
    else:
        user_config = {}

model_config = OrderedDict(model_config)
user_config = OrderedDict(user_config)
