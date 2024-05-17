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
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
CHATGLM_V3_PROMPT_FORMAT = "<|user|>\n{prompt}\n<|assistant|>"

long_input = """多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命的，”吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”--丈夫回答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫芦。
三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。“在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府。
　请详细描述霍·阿·布恩蒂亚是如何是怎样从这片崭新的天地寻找金子的？吉卜赛人带来了哪些神奇的东西？"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="THUDM/chatglm3-6b",
                        help='The huggingface repo id for the ChatGLM3 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    # Load model in optimized bf16 here.
    # Set `speculative=True`` to enable speculative decoding,
    # it only works when load_in_low_bit="fp16" on Intel CPU or load_in_low_bit="bf16" on latest Intel Xeon CPU
    model = AutoModel.from_pretrained(model_path,
                                      optimize_model=True,
                                      torch_dtype=torch.bfloat16,
                                      load_in_low_bit="bf16",
                                      speculative=True,
                                      trust_remote_code=True,
                                      use_cache=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with torch.inference_mode():
        prompt = CHATGLM_V3_PROMPT_FORMAT.format(prompt=args.prompt)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        actual_in_len = input_ids.shape[1]
        print("actual input_ids length:" + str(actual_in_len))
        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                do_sample=False,
                                attention_mask=attention_mask,
                                th_stop_draft=0.6)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                do_sample=False,
                                attention_mask=attention_mask,
                                th_stop_draft=0.6)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        end = time.perf_counter()

        print(f"E2E Generation time {(end - st):.4f}s")
        print(output_str)

        # When the IPEX_CPU optimized models recive short prompts(length < 256)
        # it will use normal generate() and has not these attr
        from ipex_llm.transformers.convert import get_enable_ipex
        _enable_ipex = get_enable_ipex()
        if not _enable_ipex or actual_in_len >= 256:
            print(f"Tokens generated {model.n_token_generated}")
            print(f"First token latency {model.first_token_time:.4f}s")
