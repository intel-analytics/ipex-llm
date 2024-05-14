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
# https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/utils/prompt_files.py
# which is licensed under Apache License 2.0
#
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

PROMPTFILE_PREFIX = 'file::'


def load_prompts(prompts: List[str],
                 prompt_delimiter: Optional[str] = None) -> List[str]:
    """Loads a set of prompts, both free text and from file.

    Args:
        prompts (List[str]): List of free text prompts and prompt files
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    prompt_strings = []
    for prompt in prompts:
        if prompt.startswith(PROMPTFILE_PREFIX):
            prompts = load_prompts_from_file(prompt, prompt_delimiter)
            prompt_strings.extend(prompts)
        else:
            prompt_strings.append(prompt)
    return prompt_strings


def load_prompts_from_file(prompt_path: str,
                           prompt_delimiter: Optional[str] = None) -> List[str]:
    """Load a set of prompts from a text fie.

    Args:
        prompt_path (str): Path for text file
        prompt_delimiter (Optional str): Delimiter for text file
            If not provided, assumes the prompt file is a single prompt (non-delimited)

    Returns:
        List of prompt string(s)
    """
    if not prompt_path.startswith(PROMPTFILE_PREFIX):
        raise ValueError(f'prompt_path_str must start with {PROMPTFILE_PREFIX}')

    _, prompt_file_path = prompt_path.split(PROMPTFILE_PREFIX, maxsplit=1)
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.')

    with open(prompt_file_path, 'r') as f:
        prompt_string = f.read()

    if prompt_delimiter is None:
        return [prompt_string]
    return [i for i in prompt_string.split(prompt_delimiter) if i]
