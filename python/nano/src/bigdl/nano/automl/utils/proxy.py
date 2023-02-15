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

import functools
from bigdl.nano.utils.common import invalidInputError


def proxy_method(cls, name):
    # This unbound method will be pulled from the superclass.
    invalidInputError(hasattr(cls, name),
                      f"%s should have %s attribute" %
                      (str(cls.__name__), name))
    proxyed = getattr(cls, name)

    @functools.wraps(proxyed)
    def wrapper(self, *args, **kwargs):
        return self._proxy(name, proxyed.__get__(self, cls), *args, **kwargs)
    return wrapper


def proxy_methods(cls):
    for name in cls.PROXYED_METHODS:
        if hasattr(cls, name):
            setattr(cls, name, proxy_method(cls, name))
    return cls
