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

class LazyImport(object):
    """Lazy import python module till use
       Args:
           module_name (string): The name of module imported later
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            # __import__ returns top level module
            top_level_module = __import__(self.module_name)
            if len(self.module_name.split('.')) == 1:
                self.module = top_level_module
            else:
                # for cases that input name is foo.bar.module
                module_list = self.module_name.split('.')
                temp_module = top_level_module
                for temp_name in module_list[1:]:
                    temp_module = getattr(temp_module, temp_name)
                self.module = temp_module

        return getattr(self.module, name)