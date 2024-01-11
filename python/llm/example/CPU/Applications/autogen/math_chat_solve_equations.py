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
# This file is adapted from https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb
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
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

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
    "request_timeout": 600,
    "seed": 45,  # change the seed for different trials
    "config_list": config_list,
    "temperature": 0,
    "max_tokens":16000,
}

# 1. create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant", 
    system_message="You are a helpful assistant.",
    llm_config=llm_config
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
# By default, the human_input_mode is "NEVER", which means the agent will not ask for human input.
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent", 
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": False},
)

# given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
# the assistant receives the message and generates a response. The response will be sent back to the mathproxyagent for processing.
# The conversation continues until the termination condition is met, in MathChat, the termination condition is the detect of "\boxed{}" in the response.
math_problem = "For what negative value of $k$ is there exactly one solution to the system of equations \\begin{align*}\ny &= 2x^2 + kx + 6 \\\\\ny &= -x + 4?\n\\end{align*}"
mathproxyagent.initiate_chat(assistant, problem=math_problem)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time} seconds")