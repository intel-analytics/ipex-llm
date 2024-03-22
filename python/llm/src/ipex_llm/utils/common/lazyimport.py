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

import importlib
import sys


# code adaptted from https://github.com/intel/neural-compressor/
#                    blob/master/neural_compressor/utils/utility.py#L88
class LazyImport:
    """
    Lazy import python module until use.

    Example:
        >>> from ipex_llm.utils.common import LazyImport
        >>> _convert_to_ggml = LazyImport('ipex_llm.ggml.convert._convert_to_ggml')
        >>> _convert_to_ggml(model_path, outfile_dir)
    """
    def __init__(self, module_name: str):
        """
        :param module_name: Import module name.
        """
        self.module_name = module_name

    def __getattr__(self, name):
        absolute_name = importlib.util.resolve_name(self.module_name)
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

        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function(*args, **kwargs)
