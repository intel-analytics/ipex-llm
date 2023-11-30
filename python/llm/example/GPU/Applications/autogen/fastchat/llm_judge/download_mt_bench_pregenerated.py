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
# Copyright 2023 The FastChat team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Download the pre-generated model answers and judgments for MT-bench.
"""
import os

from fastchat.utils import run_cmd

filenames = [
    "data/mt_bench/model_answer/alpaca-13b.jsonl",
    "data/mt_bench/model_answer/baize-v2-13b.jsonl",
    "data/mt_bench/model_answer/chatglm-6b.jsonl",
    "data/mt_bench/model_answer/claude-instant-v1.jsonl",
    "data/mt_bench/model_answer/claude-v1.jsonl",
    "data/mt_bench/model_answer/dolly-v2-12b.jsonl",
    "data/mt_bench/model_answer/falcon-40b-instruct.jsonl",
    "data/mt_bench/model_answer/fastchat-t5-3b.jsonl",
    "data/mt_bench/model_answer/gpt-3.5-turbo.jsonl",
    "data/mt_bench/model_answer/gpt-4.jsonl",
    "data/mt_bench/model_answer/gpt4all-13b-snoozy.jsonl",
    "data/mt_bench/model_answer/guanaco-33b.jsonl",
    "data/mt_bench/model_answer/guanaco-65b.jsonl",
    "data/mt_bench/model_answer/h2ogpt-oasst-open-llama-13b.jsonl",
    "data/mt_bench/model_answer/koala-13b.jsonl",
    "data/mt_bench/model_answer/llama-13b.jsonl",
    "data/mt_bench/model_answer/mpt-30b-chat.jsonl",
    "data/mt_bench/model_answer/mpt-30b-instruct.jsonl",
    "data/mt_bench/model_answer/mpt-7b-chat.jsonl",
    "data/mt_bench/model_answer/nous-hermes-13b.jsonl",
    "data/mt_bench/model_answer/oasst-sft-4-pythia-12b.jsonl",
    "data/mt_bench/model_answer/oasst-sft-7-llama-30b.jsonl",
    "data/mt_bench/model_answer/palm-2-chat-bison-001.jsonl",
    "data/mt_bench/model_answer/rwkv-4-raven-14b.jsonl",
    "data/mt_bench/model_answer/stablelm-tuned-alpha-7b.jsonl",
    "data/mt_bench/model_answer/tulu-30b.jsonl",
    "data/mt_bench/model_answer/vicuna-13b-v1.3.jsonl",
    "data/mt_bench/model_answer/vicuna-33b-v1.3.jsonl",
    "data/mt_bench/model_answer/vicuna-7b-v1.3.jsonl",
    "data/mt_bench/model_answer/wizardlm-13b.jsonl",
    "data/mt_bench/model_answer/wizardlm-30b.jsonl",
    "data/mt_bench/model_judgment/gpt-4_single.jsonl",
    "data/mt_bench/model_judgment/gpt-4_pair.jsonl",
]


if __name__ == "__main__":
    prefix = "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"

    for name in filenames:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        ret = run_cmd(f"wget -q --show-progress -O {name} {prefix + name}")
        assert ret == 0
