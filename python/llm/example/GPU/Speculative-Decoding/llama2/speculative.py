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
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
LLAMA2_PROMPT_FORMAT = """### HUMAN:
[inst]{prompt}[/inst]

### RESPONSE:
"""

long_input = """In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary.
In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined.
One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms.
In fact, it was hard to find anything in people's lives that wasn't touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve.
Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future.
Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone.
One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation.
Maya's work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts.
One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say.
As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility.
When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage.
As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives.
Overall, Maya's speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all. 
As Maya left the stage and made her way back to her seat, she couldn't help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work. 
For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone.
As she """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--precision', type=str, default='bf16',
                        help='Main model Precision')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--max-draft', type=int, default=8,
                        help='Max draft')
    parser.add_argument('--xpu', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Model temperature')
    parser.add_argument('--th_stop_draft', type=float, default=0.8,
                        help='draft stop probility')

    args = parser.parse_args()
    device = 'xpu'
    model_path = args.repo_id_or_model_path
    max_step_draft = args.max_draft

    draft_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       load_in_4bit=True,
                                                       optimize_model=True,
                                                       trust_remote_code=True)
    draft_model = draft_model.half().to(device)

    print("Assistant model loaded!")

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 torch_dtype=torch.float16,
                                                 load_in_low_bit="fp16",
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.to(device)
    print("Target model loaded!")

    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    print(f"Model is {model.dtype}")
    print(f"Max Draft number {max_step_draft}")
    print(f"Max token number {args.n_predict}")

    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

        output = model.generate(input_ids,
                                speculative=True,
                                draft_model=draft_model,
                                max_new_tokens=args.n_predict,
                                max_step_draft=max_step_draft,
                                do_sample=False,
                                th_stop_draft=args.th_stop_draft)
        output_str = tokenizer.decode(output[0])
        print(output_str)
        for i in range(2):
            st = time.perf_counter()
            output = model.generate(input_ids,
                                    speculative=True,
                                    draft_model=draft_model,
                                    max_new_tokens=args.n_predict,
                                    max_step_draft=max_step_draft,
                                    do_sample=False,
                                    th_stop_draft=args.th_stop_draft)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            if args.xpu:
                torch.xpu.synchronize()
            end = time.perf_counter()
            print("=======================================")
            print(output_str)
            print(f"Final token number {model.n_token_generated}")
            print(f"Average Draft time {sum(model.draft_time)/model.n_drafted}")
            print(f"Average Verify time {sum(model.verify_time)/len(model.verify_time)}")
            print(f"Average Generation time {sum(model.generate_time)/len(model.generate_time)}")
            print(f"Generation throughput {1.0 * (model.n_token_generated - 1) / sum(model.generate_time)}")
            print(f"E2E Generation throughput without first token {1.0 * (model.n_token_generated - 1) / model.e2e_time_without_first }")
            print(f"E2E Generation throughput {1.0 * (model.n_token_generated - 1) / (end - st) }")
            print(f"Draft num {model.n_drafted}")
            print(f"Accept num {model.n_matched}")
            print(f"Draft {model.draft_num}")
            print(f"Accept {model.accept_num}")
            print(f"Iters: {len(model.draft_num)}")
            print(f"Draft len: {model.n_drafted/len(model.draft_num)}, accept len: {model.n_matched/len(model.accept_num)}")
            print(f"Accept rate: {model.n_matched/model.n_drafted}")