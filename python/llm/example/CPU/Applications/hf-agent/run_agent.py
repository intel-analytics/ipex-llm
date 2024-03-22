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

import argparse

from PIL import Image
from transformers import AutoTokenizer, LocalAgent

from ipex_llm.transformers import AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent using vicuna model")
    parser.add_argument("--repo-id-or-model-path", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="The huggingface repo id for the Vicuna model to be downloaded"
                             ", or the path to the huggingface checkpoint folder")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Image to generate caption")

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load image
    image = Image.open(args.image_path)
    # Create an agent
    agent = LocalAgent(model, tokenizer)

    # Generate results
    prompt = "Generate a caption for the 'image'"
    print(f"Image path: {args.image_path}")
    print('==', 'Prompt', '==')
    print(prompt)
    print(agent.run(prompt, image=image))
