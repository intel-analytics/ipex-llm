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


import torch
import time
import argparse
from ipex_llm.transformers.npu_pipeline_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default=r"C:\\Llama2-converted-weights\\",
        help="The folder path of converted model blobs",
    )
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-output-len", type=int, default=1024)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 ov_model=True,
                                                 max_output_len=args.max_output_len,
                                                 model_name="Model70")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    DEFAULT_SYSTEM_PROMPT = """\
    """

    print("-" * 80)
    print("done")
    with torch.inference_mode():
        print("finish to load")
        prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        _input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print("input length:", len(_input_ids[0]))
        st = time.time()
        output = model.generate(
            _input_ids, max_new_tokens=args.n_predict,
        )
        end = time.time()
        print(f"Inference time: {end-st} s")
        input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
        print("-" * 20, "Input", "-" * 20)
        print(input_str)
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        print("-" * 20, "Output", "-" * 20)
        print(output_str)

    print("-" * 80)
    print("done")
    print("success shut down")
