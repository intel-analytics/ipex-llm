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
import importlib


def deprecated(message=""):
    def deprecated_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("{} will be deprecated in future release. {}"
                          .format(function.__name__, message),
                          category=DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            return function(*args, **kwargs)
        return wrapped
    return deprecated_decorator


class LazyImport:
    """
        :param module_name: The name of module imported later
        :param pkg: prefix path.
    """
    def __init__(self, module_name, pkg=None):
        self.module_name = module_name
        self.pkg = pkg

    def __getattr__(self, name):
        try:
            module = importlib.import_module(self.module_name, package=self.pkg)
            mod = getattr(module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + '.' + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        function_name = self.module_name.split('.')[-1]
        module_name = self.module_name.split(f'.{function_name}')[0]
        module = importlib.import_module(module_name, package=self.pkg)
        function = getattr(module, function_name)
        return function(*args, **kwargs)
