import requests
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
from tqdm import tqdm
import json
import random
import sys

if len(sys.argv) < 3:
    print("Usage: python bench.py <model> <max_seq>")
    sys.exit(1)

print("running bench.py")
model_name = sys.argv[1]
print("model_name: " + str(model_name))
max_seq = sys.argv[2]
print("max_seq: " + str(max_seq))

PROMPT_32 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

PROMPT_128 = "In a distant future, humanity has expanded across the galaxy, establishing colonies on numerous planets. The interstellar community thrives under the guidance of the United Galactic Federation, which ensures peace and prosperity. However, a new threat emerges from the unknown regions of space, challenging the stability and security of the galaxy. Brave explorers and seasoned warriors must unite to uncover the secrets of this mysterious force and protect the future of all sentient beings.  Please continue the above story as long as possible, preferably more than 1000 tokens."

PROMPT_1024 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. However, her parents were always telling her to stay close to home, to be careful, and to avoid any danger. But the little girl was stubborn, and she wanted to see what was on the other side of the mountain. So she sneaked out of the house one night, leaving a note for her parents, and set off on her journey. As she climbed the mountain, the little girl felt a sense of excitement and wonder. She had never been this far away from home before, and she couldnt wait to see what she would find on the other side. She climbed higher and higher, her lungs burning from the thin air, until she finally reached the top of the mountain. And there, she found a beautiful meadow filled with wildflowers and a sparkling stream. The little girl danced and played in the meadow, feeling free and alive. She knew she had to return home eventually, but for now, she was content to enjoy her adventure. As the sun began to set, the little girl reluctantly made her way back down the mountain, but she knew that she would never forget her adventure and the joy of discovering something new and exciting. And whenever she felt scared or unsure, she would remember the thrill of climbing the mountain and the beauty of the meadow on the other side, and she would know that she could face any challenge that came her way, with courage and determination. She carried the memories of her journey in her heart, a constant reminder of the strength she possessed. The little girl returned home to her worried parents, who had discovered her note and anxiously awaited her arrival. They scolded her for disobeying their instructions and venturing into the unknown. But as they looked into her sparkling eyes and saw the glow on her face, their anger softened. They realized that their little girl had grown, that she had experienced something extraordinary. The little girl shared her tales of the mountain and the meadow with her parents, painting vivid pictures with her words. She spoke of the breathtaking view from the mountaintop, where the world seemed to stretch endlessly before her. She described the delicate petals of the wildflowers, vibrant hues that danced in the gentle breeze. And she recounted the soothing melody of the sparkling stream, its waters reflecting the golden rays of the setting sun. Her parents listened intently, captivated by her story. They realized that their daughter had discovered a part of herself on that journey—a spirit of curiosity and a thirst for exploration. They saw that she had learned valuable lessons about independence, resilience, and the beauty that lies beyond ones comfort zone. From that day forward, the little girls parents encouraged her to pursue her dreams and embrace new experiences. They understood that while there were risks in the world, there were also rewards waiting to be discovered. They supported her as she continued to embark on adventures, always reminding her to stay safe but never stifling her spirit. As the years passed, the little girl grew into a remarkable woman, fearlessly exploring the world and making a difference wherever she went. The lessons she had learned on that fateful journey stayed with her, guiding her through challenges and inspiring her to live life to the fullest. And so, the once timid little girl became a symbol of courage and resilience, a reminder to all who knew her that the greatest joys in life often lie just beyond the mountains we fear to climb. Her story spread far and wide, inspiring others to embrace their own journeys and discover the wonders that awaited them. In the end, the little girls adventure became a timeless tale, passed down through generations, reminding us all that sometimes, the greatest rewards come to those who dare to step into the unknown and follow their hearts. With each passing day, the little girls story continued to inspire countless individuals, igniting a spark within their souls and encouraging them to embark on their own extraordinary adventures. The tale of her bravery and determination resonated deeply with people from all walks of life, reminding them of the limitless possibilities that awaited them beyond the boundaries of their comfort zones. People marveled at the little girls unwavering spirit and her unwavering belief in the power of dreams. They saw themselves reflected in her journey, finding solace in the knowledge that they too could overcome their fears and pursue their passions. The little girl\'s story became a beacon of hope, a testament to the human spirit"

PROMPT_2048 = "“You’re an idiot,” she said.\nI smiled and leaned back in the chair, looking at her over my glasses. “No, I’m not.”\n“If you were smart you would have learned to dance years ago. You’ve got two left feet.” She held up both of her hands with four fingers extended then made a circular motion that looked like an airplane.\nI leaned forward and put my glasses on the table in front of me, reaching for her hands as I did so, grabbing them before they could leave mine. “The next time you do something like this, call me. The phone number is right here,” I said as I pointed at a piece of paper under a stack of papers on my desk.\n“Fine,” she huffed and turned to leave the room. But she stopped at the doorway when she saw the bookshelves that lined one wall. “What are these for?” She stepped closer, tilting her head back and forth as she looked up. The shelves were three stories high with stacks of books on every level.\n“Books.” I smiled again. “I have a lot of books.”\nShe didn’t respond to that so I continued: “And there are more in the basement.”\n“But you can’t move them all here, right? This place is just too small for all those books. Maybe we should look for a bigger office building.” She looked back at me but said nothing as she took another few steps towards the door and then stopped again when she saw my grandfather clock on the wall.\n“And this?” she pointed to the clock, which had been in the family for over seventy years. “It’s just a clock isn’t it?”\nI laughed. “You can say that, but I know better.” It was then that I told her my grandfather’s story. He made that clock, and it was his favorite possession. When he died she inherited the clock; or at least she thought she did. After a few weeks of trying to sell it on eBay, she gave up because no one would pay what she felt it was worth.\n“You should have had an auction,” she suggested, leaning in towards me again. “Then maybe you could get more for it.”\n“No,” I shook my head. “I don’t want to sell the clock.”\nShe smiled, but this time it didn’t reach her eyes. She took a step back and looked at me again, not saying anything, just staring. The only sound was the ticking of the grandfather clock in the background as she waited for my next words.\n“My grandfather made this clock. He did everything by hand.” I could see that she had no idea what to say or do so I continued: “It’s his favorite possession, and it means more to me than anything else he ever owned. So, if you want the books, you can have them…” I looked at her face for just a second before continuing, “but you won’t take the clock.”\nShe finally responded with: “But what about the money?” She looked around again and said, “I think we could make more selling these books than you would get from all of them. You must have thousands of books here!”\nI took another step forward and put my hand on her shoulder as I spoke to her in a very low voice. “You’ve got it all wrong,” I told her. “There are only two or three hundred books. I’m not looking for money – I’m looking for someone who understands how important this clock is.”\n“How much do you want for the books?” she asked, still staring at me intently as she waited for my answer.\n“Forget about the money,” I said again. “If you really want to buy them, we can take our time and talk more later. But if you just want their value in paperbacks, that’s not what they’re worth.” She still seemed confused by everything I had said so far, so I tried to simplify my words as much as possible: “The books are mine; the clock is my grandfather’s. These books have been passed down through several generations of our family and are irreplaceable. Do you understand?”\n“I guess not,” she answered as she walked away from me, still looking at me but not saying a word. She took two more steps before turning around to say one last thing: “Well, good luck with the books, then.” With that, she went back into her house and out of sight, still walking without talking.\nAfter a few minutes, I slowly walked back toward my grandfather’s home. As I got closer, I could see the roof in the distance; the white crosses on the top of it were hard to miss. It seemed as if the entire town had gathered around there at that moment – people were all over the road around us, watching the commotion and chattering about what was going on.\nWhen my grandfather first saw me, he looked up from his chair with a smile on his face: “There you are.” He looked down at his hands, then back toward me as I walked forward to talk to him for the first time in years: “It’s been too long since we last spoke; it’s good to see you again.”\n“And you,” I said. Then, looking past my grandfather and directly into the face of the man who was sitting next to him (my mother’s father), I said, “I see he got your clock back for you, too. How is he?” My grandfather smiled as he looked up at me again:\n“He’s fine,” he answered, still smiling as he watched my mother’s family and mine chat with one another in the middle of all these people – a situation that I had never seen before. “Come on inside.” He stood up from his chair to do just that; my mom and her sister were already walking out of the building. “I have things for you.”\nMy grandfather led us inside, down some steps where he used to serve as the pastor in his church; there was a big room full of chairs at the bottom with pictures on the wall – all kinds of pictures, from when my family first started coming here to visit and other pictures we took while staying here over the years. All these photographs were all around us as I followed my grandfather through the building:\n“My house is just up the street,” he said. He stopped at a picture on the wall that was taken in the summer when we came to visit, smiling as he looked toward it with his arms folded – the picture was of him and his wife and two of their daughters, all standing together by one of the trees outside; there were other pictures around this one, some from much earlier than when my grandfather first started serving here. “We used to sit in a booth in that restaurant right over there – you remember?” I nodded as we went past it.\nMy grandfather stopped at another picture on the wall: it was of him and his wife with two other families, all sitting around a table together, smiling. He looked down at this one for a moment; then he said, “We used to do things like this every year, when we came to visit.” It was an older picture than the last one my grandfather had stopped in front of; I didn’t know it before but now I realized how much he has aged.\nMy grandparents have lived together for many years. They used to live in a house right next door, so they could walk over whenever they wanted; that is what they have done here all these years – as my grandfather said, “we’ve come here every summer since I was eleven.” But he and his wife are getting old now. He isn’t able to walk much anymore, but it makes him happy when he does: “My health has not been good lately,” he said.\n“You will never have a better time in your life than this one right now; you will never be as happy as you are now.” And for the first time since I have known him – since I was very little and started coming here every summer – my grandfather smiled at me, his eyes sparkling with excitement.\n“I know,” I said. “That’s why I’m really looking forward to it. It will be a lot of fun.” Then he turned back to the picture again; “See this?” he asked, pointing. “I remember that day, all sixteen of us there together. I was eleven then – my dad had taken me and my brother for our first trip away from home – and that was when we used to go to the cottage.” He stared at it for a while longer; he had tears in his eyes. “I loved this picture,” he said, turning it over again with one hand so I could see the back of it.\n“This is my best memory,” he explained. “It was taken on my birthday. That’s what makes me happiest.” He pointed to a man who had a pipe in his mouth. “That’s my uncle,” he said. “He gave all of us kids cigars for our birthdays, and we used to take turns lighting them – then everyone would sit around outside in the sunshine and smoke together like that. It was such a good time.” Then he held up his hand, as if to say, that’s enough now; and he went on, “Anyway, I don’"

from typing import AsyncGenerator, List, Tuple
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
def sample_requests(
    dataset_path: str,
    num_requests: int,
    model_path: str,
    seed=42
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

# 定义一个函数来执行单个请求
def perform_request(session, url, payload, headers):
    start_time = time.perf_counter()
    with session.post(url, json=payload, headers=headers, stream=True) as response:
        # 确保响应成功
        response.raise_for_status()

        # 开始接收streaming响应
        first_token_time = None
        last_token_time = 0
        first_token_inference_time = None
        next_token_inference_time = None
        next_token_time = []
        i = 0
        for line in response.iter_lines():

            token_time = time.perf_counter() - start_time
            if line:  # 忽略心跳
                data = line.decode('utf-8').strip()
                if data.startswith('data: '):
                    data = data[len('data: '):]
                    i = i + 1
                    # print(i, " ", data)
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            choice = json_data['choices'][0]
                            if 'finish_reason' in choice and (choice['finish_reason'] == 'length' or choice['finish_reason'] == 'stop'):
                                if 'first_token_time' in choice and isinstance(choice['first_token_time'], float):
                                    first_token_inference_time = choice['first_token_time']
                                if 'rest_token_time' in choice and isinstance(choice['rest_token_time'], float):
                                    next_token_inference_time = choice['rest_token_time']
                            else:
                                # 记录第一个token的时间
                                if first_token_time is None:
                                    first_token_time = token_time
                                else:
                                    # 记录后续token的时间
                                    next_token_time.append(token_time - last_token_time)
                                last_token_time = token_time
                    except json.JSONDecodeError:
                        pass  # 如果不是JSON数据，忽略错误

        # 返回第一个token和后续token的latency
        end_time = time.perf_counter()
        # print("length: ", len(next_token_time))

        return first_token_time, np.mean(next_token_time), end_time - start_time, first_token_inference_time, next_token_inference_time

def extend_list_to_length(lst, target_length):
    if target_length <= len(lst):
        return lst[:]

    # 计算每个元素需要复制的次数
    times = target_length // len(lst)
    # 计算不能整除的剩余部分
    remainder = target_length % len(lst)

    # 生成新列表：重复整个列表times次，再加上前remainder个元素
    extended_list = lst * times + lst[:remainder]

    return extended_list

# 定义一个函数来执行benchmark
def benchmark(llm_urls, model, prompt, num_requests, max_concurrent_requests, max_tokens, is_warmup=False, dataset=None):
    # 定义请求的payload和headers

    headers = {"Content-Type": "application/json"}

    first_token_latencies = []
    next_token_latencies = []
    total_responce_times = []
    first_token_inference_times = []
    next_token_inference_times = []
    cur_url_index = 0
    sampled_requests = []
    prompt_token_lens = []
    output_tokens_lens = []


    if not dataset is None:
        sampled_requests = sample_requests(dataset, num_requests, model)

    # 使用Session对象以便复用连接
    with requests.Session() as session:
        # 创建一个线程池
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # 开始计时
            # time.sleep(1)
            llm_url = llm_urls[cur_url_index]
            cur_url_index = (cur_url_index + 1) % len(llm_urls)

            cur_llm_urls = extend_list_to_length(llm_urls, max_concurrent_requests)
            cur_len = len(cur_llm_urls)
            if dataset is None:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "ignore_eos": True,
                    "stream": True  # 开启streaming模式
                }
                futures = [executor.submit(perform_request, session, cur_llm_urls[index % cur_len], payload, headers) for index in range(num_requests)]
            else:
                payloads = []
                for index in range(num_requests):
                    prompt, prompt_len, output_len = sampled_requests[index]
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "n": 1,
                        "best_of": 1,
                        "use_beam_search": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": output_len,
                        "ignore_eos": True,
                        "stream": True  # 开启streaming模式
                    }
                    prompt_token_lens.append(prompt_len)
                    output_tokens_lens.append(output_len)
                    payloads.append(payload)
                futures = [executor.submit(perform_request, session, cur_llm_urls[index % cur_len], payloads[index], headers) for index in range(num_requests)]


            start_time = time.perf_counter()

            if is_warmup:
                phase = "Warm Up"
            else:
                phase = "Benchmarking"
            with tqdm(total=num_requests, desc=phase, unit="req", ncols=100) as pbar:
                # 等待所有请求完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        first_token_latency, next_token_latency, total_responce_time, first_token_inference_time, next_token_inference_time = future.result()
                        first_token_latencies.append(first_token_latency)
                        next_token_latencies.append(next_token_latency)
                        total_responce_times.append(total_responce_time)
                        if first_token_inference_time:
                            first_token_inference_times.append(first_token_inference_time)
                        if next_token_inference_time:
                            next_token_inference_times.append(next_token_inference_time)
                    except Exception as e:
                        print(f"Request failed: {e}")
                    pbar.update(1)

            # 计算总用时
            if is_warmup:
                return
            total_time = time.perf_counter() - start_time
            print(f"Total time for {num_requests} requests with {max_concurrent_requests} concurrent requests: {total_time} seconds.")
            print(f"Average responce time: {np.mean(total_responce_times)}")
            if dataset is None:
                print(f"Token throughput: {num_requests * max_tokens / total_time}")
            else:
                print(f"Output token throughput: {sum(output_tokens_lens) / total_time}")
                print(f"Total token throughput: {(sum(prompt_token_lens) + sum(output_tokens_lens)) / total_time}")
            print()
            if first_token_latencies:
                average_first_token_latency = sum(first_token_latencies) / len(first_token_latencies)
                p90_first_token_latency = np.percentile(first_token_latencies, 90)
                p95_first_token_latency = np.percentile(first_token_latencies, 95)
                average_first_token_inference_latency = np.mean(first_token_inference_times)
                print(f"Average first token latency: {average_first_token_latency * 1000} milliseconds.")
                print(f"P90 first token latency: {p90_first_token_latency * 1000} milliseconds.")
                print(f"P95 first token latency: {p95_first_token_latency * 1000} milliseconds.")
                #print(f"Average first token inference latency: {average_first_token_inference_latency * 1000} milliseconds.")
                print()
            if next_token_latencies:
                average_next_token_latency = sum(next_token_latencies) / len(next_token_latencies)
                p90_next_token_latency = np.percentile(next_token_latencies, 90)
                p95_next_token_latency = np.percentile(next_token_latencies, 95)
                average_next_token_inference_latency = np.mean(next_token_inference_times)
                print(f"Average next token latency: {average_next_token_latency * 1000} milliseconds.")
                print(f"P90 next token latency: {p90_next_token_latency * 1000} milliseconds.")
                print(f"P95 next token latency: {p95_next_token_latency * 1000} milliseconds.")
                #print(f"Average next token inference latency: {average_next_token_inference_latency * 1000} milliseconds.")
                print()


# 设置benchmark参数
LLM_URLS = [f"http://localhost:{PORT}/v1/completions" for PORT in [8000]]


MODEL = "/llm/models/" + model_name
MAX_TOKENS = 512

PROMPT = PROMPT_1024

max_batch=int(max_seq)

for MAX_CONCURRENT_REQUESTS in [max_batch]:
    NUM_WARMUP = 2 * MAX_CONCURRENT_REQUESTS
    NUM_REQUESTS = 5 * MAX_CONCURRENT_REQUESTS  # 总请求次数

    # to avoid warm_up time out
    benchmark(LLM_URLS, MODEL, PROMPT_1024, 2, 1, 32, is_warmup = True)
    benchmark(LLM_URLS, MODEL, PROMPT, NUM_WARMUP, MAX_CONCURRENT_REQUESTS, MAX_TOKENS, is_warmup = True)

    # 运行benchmark
    benchmark(LLM_URLS, MODEL, PROMPT, NUM_REQUESTS, MAX_CONCURRENT_REQUESTS, MAX_TOKENS)
