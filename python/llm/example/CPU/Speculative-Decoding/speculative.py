import torch
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer
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
long_input_cn = """多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命的，”吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”--丈夫回答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫芦。
三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。“在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府。
　请详细描述霍·阿·布恩蒂亚是如何是怎样从这片崭新的天地寻找金子的？吉卜赛人带来了哪些神奇的东西？"""

# Prompt format dictionary
PROMPT_FORMATS = {
    'chatglm': "\n{prompt}\n",
    'llama': """### HUMAN:
[inst]{prompt}[/inst]

### RESPONSE:
""",
    'baichuan': "<human>{prompt} <bot>",
    'mistral': "{prompt}",  # Custom format for Mistral model
    'qwen': """
system
You are a helpful assistant.

user
{prompt}

assistant
""",  # Format for Qwen model
    'vicuna': "### Human:\n{prompt} \n ### Assistant:\n"  # Format for Vicuna model
}


# Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for models')
    parser.add_argument('--model-type', type=str, default="llama",
                        help='Type of the model to use (chatglm, llama, baichuan, mistral, qwen, vicuna)')
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
    if model_type in ['chatglm', 'baichuan', 'mistral', 'qwen', 'vicuna']:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     torch_dtype=torch.bfloat16,
                                                     load_in_low_bit="bf16",
                                                     speculative=True,
                                                     torchscript=True,
                                                     trust_remote_code=True,
                                                     use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == 'llama':
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     torch_dtype=torch.bfloat16,
                                                     load_in_low_bit="bf16",
                                                     speculative=True,
                                                     torchscript=True,
                                                     trust_remote_code=True,
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        raise ValueError("Unsupported model type. Choose from 'chatglm', 'llama', 'baichuan', 'mistral', 'qwen', or 'vicuna'.")

    # Run the model
    with torch.inference_mode():
        prompt = prompt_format.format(prompt=args.prompt)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        actual_in_len = input_ids.shape[1]
        print("actual input_ids length:" + str(actual_in_len))

        # Warm-up and speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                attention_mask=attention_mask,
                                do_sample=False,
                                th_stop_draft=args.th_stop_draft)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        end = time.perf_counter()

        print(output_str)
        print(f"Tokens generated {model.n_token_generated}")
        print(f"E2E Generation time {(end - st):.4f}s")
        print(f"First token latency {model.first_token_time:.4f}s")
