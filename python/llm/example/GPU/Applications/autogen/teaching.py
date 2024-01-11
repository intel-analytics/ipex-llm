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
# This file is adapted from https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teaching.ipynb
# which is licensed under Apache License 2.0
#
# Copyright 2023 The AutoGen team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import time
import autogen

start_time = time.time()

config_list = [
    {
        "model": "Llama-2-7b-chat-hf",
        "api_base": "http://localhost:8000/v1",
        "api_type": "open_ai",
        "api_key": "NULL",
    }]


# configuration for autogen's enhanced inference API which is compatible with OpenAI API
llm_config = {
    "request_timeout": 1000,
    "seed": 45,  # change the seed for different trials
    "config_list": config_list,
    "temperature": 0,
    "max_tokens":16000,
}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
    code_execution_config={
        "work_dir": "work_dir",
        "use_docker": False,
    },
)

task1 = """
Find arxiv papers that show how are people studying trust calibration in AI based systems 
"""

user_proxy.initiate_chat(assistant, message=task1)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time} seconds")