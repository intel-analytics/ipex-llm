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

def extended_method(*classes):
    """A helper function to extract all extend method from base classes."""
    name_set = set()
    for extend_class in classes:
        for name in dir(extend_class):
            if not name.startswith("_"):
                name_set.add(name)
    return name_set


def override_method(target_cls, inherited_cls, f_wrapper):
    """
    Override target cls with inherited class method to \
    avoid functional usage issue from tf.keras submodule.
    For example, to make nano.tf.keras.Model inherite tf.keras.Model without functional usage issue.
    :param target_cls:      nano extended class, e.g. nano.tf.keras.Model
    :param inherited_cls:   desired inherited class from tf.keras, e.g. tf.keras.Model
    :param f_wrapper:       wrapper function to map class method
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
                    setattr(target_cls, name+"_old", old_f)
                    setattr(target_cls, name, getattr(cls, name))
