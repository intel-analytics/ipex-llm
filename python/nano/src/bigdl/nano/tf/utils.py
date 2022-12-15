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
import inspect
import operator
from functools import partial
from bigdl.nano.common.compare_version import _compare_version

KERAS_VERSION_LESS_2_9 = _compare_version("keras", operator.lt, "2.9")


class _NanoPartial(partial):
    pass


def patch_attrs(target_obj: object, source_obj: object) -> object:
    """
    Patch attributes of `source_obj` to `target_obj`.

    :param target_obj: target object
    :param source_obj: source object
    :return: `target_obj`
    """
    for name in set(dir(source_obj)) - set(dir(target_obj)):
        attr = getattr(source_obj, name)
        if inspect.isfunction(attr):            # static method
            setattr(target_obj, name, attr)
        elif inspect.ismethod(attr):            # method
            static_method = getattr(type(source_obj), name)
            setattr(target_obj, name, _NanoPartial(static_method, target_obj))
        elif isinstance(attr, _NanoPartial):     # replaced method by Nano
            static_method = attr.func
            setattr(target_obj, name, _NanoPartial(static_method, target_obj))
        else:
            setattr(target_obj, name, attr)
    return target_obj
