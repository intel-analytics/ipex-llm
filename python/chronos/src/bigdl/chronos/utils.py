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
import pickle
import hmac
import hashlib
from bigdl.nano.utils.common import invalidInputError


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


# code adaptted from https://github.com/intel/neural-compressor/
#                    blob/master/neural_compressor/utils/utility.py#L62
class LazyImport:
    """
    Example:
        >>> from bigdl.chronos.utils import LazyImport
        >>> forecaster = LazyImport('bigdl.chronos.forecaster.lstm_forecaster.LSTMForecaster')
        >>> forecaster.from_tsdataset(tsdataset, ...)
        >>> forecaster(...)
    """
    def __init__(self, module_name: str, pkg=None):
        """
        :param module_name: Import module name.
        :param pkg: prefix path.
        """
        self.module_name = module_name
        self.pkg = pkg

    def __getattr__(self, name):
        absolute_name = importlib.util.resolve_name(self.module_name, self.pkg)
        # not reload modules
        try:
            return getattr(sys.modules[absolute_name], name)
        except (KeyError, AttributeError):
            pass

        if "." in absolute_name:
            # Split module name to prevent class name from being introduced as package
            parent_name, _, child_name = absolute_name.rpartition('.')
        else:
            parent_name, child_name = absolute_name, None

        try:
            # For import parent module and get the submodule with getattr.
            module = importlib.import_module(parent_name)
            module = getattr(module, child_name) if child_name else module
        except AttributeError:
            # Triggered when the parent module cannot get the child module using getattr.
            # More common when calling staticmethods or classmethods. e.g. from_tsdataset.
            full_module_name = parent_name+'.'+child_name if child_name else parent_name
            spec = importlib.util.find_spec(full_module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        function_name = self.module_name.rpartition('.')[-1]
        module_name = self.module_name.rpartition(f'.{function_name}')[0]
        try:
            module = sys.modules[module_name]
        except KeyError:
            pass

        module = importlib.import_module(module_name, package=self.pkg)
        function = getattr(module, function_name)
        return function(*args, **kwargs)


# Refer to this guide https://www.synopsys.com/blogs/software-security/python-pickling/
# To safely use python pickle
class SafePickle:
    key = b'shared-key'
    """
    Example:
        >>> from bigdl.chronos.utils import SafePickle
        >>> with open(file_path, 'wb') as file:
        >>>     signature = SafePickle.dump(data, file, return_digest=True)
        >>> with open(file_path, 'rb') as file:
        >>>     data = SafePickle.load(file, signature)
    """
    @classmethod
    def dump(self, obj, file, return_digest=False, *args, **kwargs):
        if return_digest:
            pickled_data = pickle.dumps(obj)
            file.write(pickled_data)
            digest = hmac.new(self.key, pickled_data, hashlib.sha1).hexdigest()
            return digest
        else:
            pickle.dump(obj, file, *args, **kwargs)

    @classmethod
    def load(self, file, digest=None, *args, **kwargs):
        if digest:
            content = file.read()
            new_digest = hmac.new(self.key, content, hashlib.sha1).hexdigest()
            if digest != new_digest:
                invalidInputError(False, 'Pickle safe check failed')
            file.seek(0)
        return pickle.load(file, *args, **kwargs)
