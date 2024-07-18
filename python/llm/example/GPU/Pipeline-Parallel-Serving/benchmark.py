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

import requests
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
from tqdm import tqdm
import json
import argparse
from typing import List, Tuple


# Execute single request
def perform_request(session, url, payload, headers):
    start_time = time.perf_counter()
    with session.post(url, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        first_token_time = None
        last_token_time = 0
        first_token_inference_time = None
        next_token_inference_time = None
        next_token_time = []
        i = 0
        for line in response.iter_lines():
            token_time = time.perf_counter() - start_time
            if line:
                data = line.decode('utf-8').strip()
                if data.startswith('data: '):
                    data = data[len('data: '):]
                    i = i + 1
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
                                if first_token_time is None:
                                    first_token_time = token_time
                                else:
                                    next_token_time.append(token_time - last_token_time)
                                last_token_time = token_time
                    except json.JSONDecodeError:
                        pass
        end_time = time.perf_counter()
        return (
            first_token_time,
            np.mean(next_token_time),
            end_time - start_time,
            first_token_inference_time,
            next_token_inference_time,
        )


def extend_list_to_length(lst, target_length):
    if target_length <= len(lst):
        return lst[:]
    times = target_length // len(lst)
    remainder = target_length % len(lst)
    extended_list = lst * times + lst[:remainder]

    return extended_list


def benchmark(
    llm_urls,
    prompt,
    num_warmup_requests,
    num_requests,
    max_concurrent_requests,
    max_tokens,
    prompt_length,
):

    headers = {"Content-Type": "application/json"}

    first_token_latencies = []
    next_token_latencies = []
    total_responce_times = []
    first_token_inference_times = []
    next_token_inference_times = []
    cur_url_index = 0

    num_requests = num_requests + num_warmup_requests

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            llm_url = llm_urls[cur_url_index]
            cur_url_index = (cur_url_index + 1) % len(llm_urls)

            cur_llm_urls = extend_list_to_length(llm_urls, max_concurrent_requests)
            cur_len = len(cur_llm_urls)

            payload = {
                "model": "Meta-Llama-3-8B-Instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
                # for vllm openai api server
                "ignore_eos": True,
                "n": 1,
                "best_of": 1,
                "use_beam_search": False,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            futures = [
                executor.submit(
                    perform_request,
                    session,
                    cur_llm_urls[index % cur_len],
                    payload,
                    headers,
                )
                for index in range(num_requests)
            ]

            phase = "Benchmarking"

            with tqdm(total=num_requests, desc=phase, unit="req", ncols=100) as pbar:
                cur_index = 0
                for future in concurrent.futures.as_completed(futures):
                    if cur_index == num_warmup_requests:
                        start_time = time.perf_counter()
                    try:
                        (
                            first_token_latency,
                            next_token_latency,
                            total_responce_time,
                            first_token_inference_time,
                            next_token_inference_time,
                        ) = future.result()
                        cur_index = cur_index + 1
                        if cur_index > num_warmup_requests:
                            first_token_latencies.append(first_token_latency)
                            next_token_latencies.append(next_token_latency)
                            total_responce_times.append(total_responce_time)
                            if first_token_inference_time:
                                first_token_inference_times.append(
                                    first_token_inference_time
                                )
                            if next_token_inference_time:
                                next_token_inference_times.append(next_token_inference_time)
                    except Exception as e:
                        print(f"Request failed: {e}")
                    pbar.update(1)

            total_time = time.perf_counter() - start_time
            log_file = f"{max_concurrent_requests}.log"
            num_requests = num_requests - num_warmup_requests

            with open(log_file, "w") as file:
                print(
                    f"Total time for {num_requests} requests with {max_concurrent_requests} concurrent requests: {total_time} seconds.",
                    file=file,
                )
                print(
                    f"Average response time: {np.mean(total_responce_times)}", file=file
                )

                print(
                    f"Token throughput: {num_requests * max_tokens / total_time}",
                    file=file,
                )
                print(
                    f"Total token throughput: {(max_tokens + prompt_length) * num_requests / total_time}",
                    file=file,
                )
                print(file=file)

                if first_token_latencies:
                    average_first_token_latency = sum(first_token_latencies) / len(
                        first_token_latencies
                    )
                    p90_first_token_latency = np.percentile(first_token_latencies, 90)
                    p95_first_token_latency = np.percentile(first_token_latencies, 95)
                    print(
                        f"Average first token latency: {average_first_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(
                        f"P90 first token latency: {p90_first_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(
                        f"P95 first token latency: {p95_first_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(file=file)

                if next_token_latencies:
                    average_next_token_latency = sum(next_token_latencies) / len(
                        next_token_latencies
                    )
                    p90_next_token_latency = np.percentile(next_token_latencies, 90)
                    p95_next_token_latency = np.percentile(next_token_latencies, 95)
                    print(
                        f"Average next token latency: {average_next_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(
                        f"P90 next token latency: {p90_next_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(
                        f"P95 next token latency: {p95_next_token_latency * 1000} milliseconds.",
                        file=file,
                    )
                    print(file=file)


LLM_URLS = [f"http://localhost:{PORT}/v1/completions" for PORT in [8000]]

parser = argparse.ArgumentParser(description="Set prompt length.")
parser.add_argument(
    "--prompt_length",
    type=int,
    choices=[32, 128, 1024, 2048],
    default=1024,
    help="Length of the prompt: 32, 1024, or 2048",
)
parser.add_argument(
    "--max_concurrent_requests",
    type=int,
    nargs="+",
    default=[1, 2, 4, 5, 6],
    help="List of maximum concurrent requests to test.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=128,
    help="Maximum number of new tokens that the model will generate per request.",
)
args = parser.parse_args()
PROMPT_LENGTH = args.prompt_length
PROMPT = open(f"prompt/{PROMPT_LENGTH}.txt", "r").read()
MAX_TOKENS = args.max_new_tokens


for MAX_CONCURRENT_REQUESTS in args.max_concurrent_requests:
    NUM_WARMUP = 5 * MAX_CONCURRENT_REQUESTS
    NUM_REQUESTS = 30 * MAX_CONCURRENT_REQUESTS

    benchmark(LLM_URLS, PROMPT, NUM_WARMUP, NUM_REQUESTS, MAX_CONCURRENT_REQUESTS, MAX_TOKENS, PROMPT_LENGTH)
