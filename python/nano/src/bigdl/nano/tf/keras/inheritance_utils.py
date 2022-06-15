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

def f_wrapper(f):
    """A wrapper function to overide desired method."""
    def wrapped(self, *args, **kwargs):
        return f(self.model, *args, **kwargs)
    return wrapped


def extended_method(*classes):
    """A helper function to extract all extend method from base classes."""
    name_set = set()
    for extend_class in classes:
        for name in dir(extend_class):
            if not name.startswith("_"):
                name_set.add(name)
    return name_set


def override_method(target_cls, inherited_cls):
    """Override target cls with inherited class method to avoid functional usage issue from tf.keras
       For example, inherited 
    """
    extend_methods = extended_method(target_cls.__bases__)
    # map all public method inherited class from tf.keras
    for name in dir(inherited_cls):
        if name not in extend_methods and not name.startswith('_'):
            setattr(target_cls, name, f_wrapper(getattr(inherited_cls, name)))

    # map all public method from base class
    for cls in target_cls.__bases__:
        for name in dir(cls):
            if not name.startswith("_"):
                setattr(target_cls, name, f_wrapper(getattr(cls, name)))

