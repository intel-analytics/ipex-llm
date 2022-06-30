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
def inject_function(target_cls, *injected_cls):
    """
    Inject function to base directly.

    :param target_cls:      nano extended class, e.g. nano.tf.keras.Model
    :param injected_cls:    class with extended method for tf base
    """
    for cls in injected_cls:
        for name in dir(cls):
            if not name.startswith("_"):
                if name in dir(target_cls):
                    old_f = getattr(target_cls, name)
                    setattr(target_cls, name + "_old", old_f)
                    setattr(target_cls, name, getattr(cls, name))
                else:
                    setattr(target_cls, name, getattr(cls, name))
