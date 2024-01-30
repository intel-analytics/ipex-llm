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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/prompts.py


from pathlib import Path

from modules.text_generation import get_encoded_length


def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    else:
        file_path = Path(f'prompts/{fname}.txt')
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]

            return text


def count_tokens(text):
    try:
        tokens = get_encoded_length(text)
        return str(tokens)
    except:
        return '0'
