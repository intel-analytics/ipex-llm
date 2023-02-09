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
import atexit
import logging
from typing import List

logger = logging.getLogger(__name__)
warning_messages: List[str] = []


@atexit.register
def output_suggestions():
    global warning_messages
    if len(warning_messages) > 0:
        print(f"\n*****************Nano performance Suggestions*****************", flush=True)
        for message in warning_messages:
            print(message, flush=True)
        print(f"\n*****************Nano performance Suggestions*****************", flush=True)


def register_suggestion(warning_message: str):
    global warning_messages

    # in case core dump / seg fault
    print(warning_message, flush=True)

    warning_messages.append(warning_message)
