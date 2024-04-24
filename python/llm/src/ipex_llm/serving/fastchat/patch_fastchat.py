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
# Some parts of this file is adapted from
# https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict, Union, Tuple

from fastchat.conversation import SeparatorStyle

def patch_get_prompt(self) -> str:
    """Get the prompt for generation."""
    system_prompt = self.system_template.format(system_message=self.system_message)
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret
    elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
        seps = [self.sep, self.sep2]
        ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ": "  # must be end with a space
        return ret
    elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
        ret = "" if system_prompt == "" else system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + "\n" + message + self.sep
            else:
                ret += role + "\n"
        return ret
    elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        ret = system_prompt
        for role, message in self.messages:
            if message:
                ret += role + message + self.sep
            else:
                ret += role
        return ret
    elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
        seps = [self.sep, self.sep2]
        ret = system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + message + seps[i % 2]
            else:
                ret += role
        return ret
    elif self.sep_style == SeparatorStyle.RWKV:
        ret = system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += (
                    role
                    + ": "
                    + message.replace("\r\n", "\n").replace("\n\n", "\n")
                )
                ret += "\n\n"
            else:
                ret += role + ":"
        return ret
    elif self.sep_style == SeparatorStyle.LLAMA2:
        seps = [self.sep, self.sep2]
        if self.system_message:
            ret = system_prompt
        else:
            ret = "[INST] "
        for i, (role, message) in enumerate(self.messages):
            tag = self.roles[i % 2]
            if message:
                if i == 0:
                    ret += message + " "
                else:
                    ret += tag + " " + message + seps[i % 2]
            else:
                ret += tag
        return ret
    elif self.sep_style == SeparatorStyle.CHATGLM:
        # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
        # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
        round_add_n = 1 if self.name == "chatglm2" else 0
        if system_prompt:
            ret = system_prompt + self.sep
        else:
            ret = ""
        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                ret += f"[Round {i//2 + round_add_n}]{self.sep}"
            if message:
                ret += f"{role}：{message}{self.sep}"
            else:
                ret += f"{role}："
        return ret
    elif self.sep_style == SeparatorStyle.CHATML:
        ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
        for role, message in self.messages:
            if message:
                ret += role + "\n" + message + self.sep + "\n"
            else:
                ret += role + "\n"
        return ret
    elif self.sep_style == SeparatorStyle.CHATGLM3:
        ret = ""
        if self.system_message:
            ret += system_prompt
        for role, message in self.messages:
            if message:
                ret += role + "\n" + message + "\n"
            else:
                ret += role + "\n"
        return ret
    elif self.sep_style == SeparatorStyle.CHATINTERN:
        # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
        seps = [self.sep, self.sep2]
        ret = system_prompt
        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                ret += "<s>"
            if message:
                ret += role + ":" + message + seps[i % 2] + "\n"
            else:
                ret += role + ":"
        return ret
    elif self.sep_style == SeparatorStyle.DOLLY:
        seps = [self.sep, self.sep2]
        ret = system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ":\n" + message + seps[i % 2]
                if i % 2 == 1:
                    ret += "\n\n"
            else:
                ret += role + ":\n"
        return ret
    elif self.sep_style == SeparatorStyle.PHOENIX:
        ret = system_prompt
        for role, message in self.messages:
            if message:
                ret += role + ": " + "<s>" + message + "</s>"
            else:
                ret += role + ": " + "<s>"
        return ret
    elif self.sep_style == SeparatorStyle.ROBIN:
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ":\n" + message + self.sep
            else:
                ret += role + ":\n"
        return ret
    elif self.sep_style == SeparatorStyle.FALCON_CHAT:
        ret = ""
        if self.system_message:
            ret += system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret
    elif self.sep_style == SeparatorStyle.METAMATH:
        ret = "" if system_prompt == "" else system_prompt + self.sep
        for i, (role, message) in enumerate(self.messages):
            # For MetaMath, sep2 is used to prefix the message.
            starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
            ending_sep = self.sep if i % 2 == 0 else ""
            if message:
                ret += role + starting_sep + message + ending_sep
            else:
                ret += role + starting_sep
        return ret
    elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
        seps = [self.sep, self.sep2]
        ret = system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        raise ValueError(f"Invalid style: {self.sep_style}")
