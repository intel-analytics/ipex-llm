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


import inspect


def get_default_args(func):
    """
    Check function `func` and get its arguments which has default value.

    :param func: Function to check.
    :return: A dict, contains arguments and their default values.
    """
    default_args = {}
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if param.default is not param.empty:
            default_args[param.name] = param.default
    return default_args
