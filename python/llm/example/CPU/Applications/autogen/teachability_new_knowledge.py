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

import autogen
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability

autogen.Completion.clear_cache()

config_list = [
    {
        "api_key": "NULL",
        
        # ----------- fastchat
        "model": "bigdl",
        "base_url": "http://localhost:8000/v1",

        # ----------- vllm 
        # "model": "hello",
        # "base_url": "http://localhost:65533/v1",
    }]

llm_config={
    "config_list": config_list,  
    "timeout": 1000, 
    "max_tokens": 256,
    "cache_seed": None, # Disable caching.
    "seed": 2024,
    "temperature": 0,
}  
    
    
# Start by instantiating any agent that inherits from ConversableAgent.
teachable_agent = ConversableAgent(
    name="teachable_agent",  # The name is flexible, but should not contain spaces to work in group chat.
    llm_config=llm_config,
)

# Instantiate the Teachability capability. Its parameters are all optional.
teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/autogen/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

# Now add the Teachability capability to the agent.
teachability.add_to_agent(teachable_agent)

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


# Instantiate a UserProxyAgent to represent the user. But in this notebook, all user input will be simulated.
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
)

text = "What is the Vicuna model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "Vicuna is a 13B-parameter language model released by Meta."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "What is the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "Orca is a 13B-parameter language model developed by Microsoft. It outperforms Vicuna on most tasks."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "How does the Vicuna model compare to the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)