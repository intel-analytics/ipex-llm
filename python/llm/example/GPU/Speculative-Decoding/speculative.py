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
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


# Reset parameter settings
torch.nn.Linear.reset_parameters = lambda x: None
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Default prompts
long_input_en = """In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary.
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
long_input_cn = """折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。问: 请基于以上描述，分析哪些步骤做错了很大可能会导致最终折叠失败？答: """


# Prompt format dictionary
PROMPT_FORMATS = {
    'chatglm': "{prompt}",
    'baichuan': "{prompt}",
    'gpt': "{prompt}",
    'mistral': "{prompt}",
    'llama': """### HUMAN:
[inst]{prompt}[/inst]

### RESPONSE:
""",
    'qwen': """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
}


# Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for models')
    parser.add_argument('--model-type', type=str, default="llama",
                        help='Type of the model to use (chatglm, llama, baichuan, mistral, qwen, gpt)')
    parser.add_argument('--repo-id-or-model-path', type=str, 
                        help='The huggingface repo id for the model to be downloaded or the path to the checkpoint folder')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--th-stop-draft', type=float, default=0.6,
                        help='Threshold for stopping draft in model generation')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    model_type = args.model_type.lower()

    # Select default input if none is provided
    if args.prompt is None:
        args.prompt = long_input_cn if model_type in ['chatglm', 'baichuan', 'qwen'] else long_input_en

    # Check supported model types and get corresponding prompt format
    if model_type not in PROMPT_FORMATS:
        raise ValueError("Unsupported model type. Supported types are: " + ", ".join(PROMPT_FORMATS.keys()))

    prompt_format = PROMPT_FORMATS[model_type]

    # Load different models and tokenizers based on model type
    if model_type in ['baichuan', 'qwen', 'gpt', 'mistral']:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    optimize_model=True,
                                                    torch_dtype=torch.float16,
                                                    load_in_low_bit="fp16",
                                                    speculative=True,
                                                    trust_remote_code=True,
                                                    use_cache=True)
        model = model.to('xpu')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == 'llama':
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    optimize_model=True,
                                                    torch_dtype=torch.float16,
                                                    load_in_low_bit="fp16",
                                                    speculative=True,
                                                    trust_remote_code=True,
                                                    use_cache=True)
        model = model.to('xpu')
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    elif model_type == 'chatglm':
        model = AutoModel.from_pretrained(model_path,
                                        optimize_model=True,
                                        torch_dtype=torch.float16,
                                        load_in_low_bit="fp16",
                                        speculative=True,
                                        trust_remote_code=True,
                                        use_cache=True)
        model = model.to('xpu')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise ValueError("Unsupported model type. Choose from 'chatglm', 'llama', 'baichuan', 'mistral', 'qwen', or 'gpt'.")

    # Run the model
    with torch.inference_mode():
        prompt = prompt_format.format(prompt=args.prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                th_stop_draft=args.th_stop_draft,
                                do_sample=False)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                th_stop_draft=args.th_stop_draft,
                                do_sample=False)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.xpu.synchronize()
        end = time.perf_counter()

        print(output_str)
        print(f"Tokens generated {model.n_token_generated}")
        print(f"E2E Generation time {(end - st):.4f}s")
        print(f"First token latency {model.first_token_time:.4f}s")
