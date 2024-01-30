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
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/block_requests.py


import builtins
import io

import requests

from modules.logging_colors import logger

original_open = open
original_get = requests.get


class RequestBlocker:

    def __enter__(self):
        requests.get = my_get

    def __exit__(self, exc_type, exc_value, traceback):
        requests.get = original_get


class OpenMonkeyPatch:

    def __enter__(self):
        builtins.open = my_open

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.open = original_open


def my_get(url, **kwargs):
    logger.info('Unwanted HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)


# Kindly provided by our friend WizardLM-30B
def my_open(*args, **kwargs):
    filename = str(args[0])
    if filename.endswith('index.html'):
        with original_open(*args, **kwargs) as f:
            file_contents = f.read()

        file_contents = file_contents.replace(b'\t\t<script\n\t\t\tsrc="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.7/iframeResizer.contentWindow.min.js"\n\t\t\tasync\n\t\t></script>', b'')
        file_contents = file_contents.replace(b'cdnjs.cloudflare.com', b'127.0.0.1')
        return io.BytesIO(file_contents)
    else:
        return original_open(*args, **kwargs)
