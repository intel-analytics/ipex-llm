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
import sys


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
        code adaptted from https://github.com/intel/neural-compressor
        /blob/master/neural_compressor/utils/utility.py#L62

        :param module_name: The name of module imported later
        :param pkg: prefix path.
    """
    def __init__(self, module_name: str, pkg=None):
        self.module_name = module_name
        self.pkg = pkg

    def __getattr__(self, name):
        abslute_name = importlib.util.resolve_name(self.module_name, self.pkg)
        # not reload modules
        try:
            return getattr(sys.modules[abslute_name], name)
        except (KeyError, AttributeError):
            pass

        try:
            module = importlib.import_module(abslute_name)
            module = getattr(module, name)
        except AttributeError:
            spec = importlib.util.find_spec(self.module_name+'.'+name)
            if spec is None:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  f"No module named {self.module_name}.")
            else:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        return module

    def __call__(self, *args, **kwargs):
        function_name = self.module_name.split('.')[-1]
        module_name = self.module_name.split(f'.{function_name}')[0]
        try:
            module = sys.modules[module_name]
        except KeyError:
            module = importlib.import_module(module_name, package=self.pkg)
        function = getattr(module, function_name)
        return function(*args, **kwargs)
