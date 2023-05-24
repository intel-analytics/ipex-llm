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


import warnings
from functools import wraps


def deprecated(func_name=None, message=""):
    def deprecated_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            funcname = function.__name__ if func_name is None else func_name
            warnings.warn("`{}` will be deprecated in future release. {}"
                          .format(funcname, message),
                          category=DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            return function(*args, **kwargs)
        return wrapped
    return deprecated_decorator
