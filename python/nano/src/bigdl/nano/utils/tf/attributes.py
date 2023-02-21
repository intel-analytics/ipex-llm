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
from functools import partial

from tensorflow.keras import Model

from bigdl.nano.utils.common import invalidOperationError


class _NanoPartial(partial):
    pass


class _ModuleWrapper:
    def __init__(self, target_obj, source_obj):
        self.__dict__["target_obj"] = target_obj
        self.__dict__["source_obj"] = source_obj
        self.__dict__["support_operations"] = ["predict", "evaluate", "compile"]

    def __getattr__(self, name):
        # We only support tf's `predict`, `evaluate`, `compile`,
        # and user custom attributes which are not in `tf.keras.Model`
        if name in self.support_operations:
            return getattr(self.target_obj, name)
        elif not hasattr(Model, name):
            try:
                return getattr(self.target_obj, name)
            except AttributeError as _e:
                pass
            return getattr(self.source_obj, name)
        else:
            invalidOperationError(False,
                                  f"This optimized model does not support {name} method !")

    def __setattr__(self, name: str, value) -> None:
        return setattr(self.target_obj, name, value)

    def __call__(self, *args, **kwargs):
        return self.target_obj(*args, **kwargs)


def patch_attrs(target_obj: object, source_obj: object) -> object:
    """
    Patch attributes of `source_obj` to `target_obj`.

    :param target_obj: target object
    :param source_obj: source object
    :return: `target_obj`
    """
    if source_obj is None:
        return target_obj
    if inspect.ismethod(target_obj.__setattr__):
        # `target_obj` has custom `__setattr__`
        wrapper_obj = _ModuleWrapper(target_obj, source_obj)
        return wrapper_obj
    else:
        # `target_obj` has no custom `__setattr__`
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


def patch_compiled(target_model: Model, source_model: Model):
    """Patch the compiled loss and metric of `source_model` to `target_model`."""
    if source_model._is_compiled:
        kwargs = {"run_eagerly": source_model._run_eagerly,
                  "steps_per_execution": int(source_model._steps_per_execution)}
        if source_model.compiled_loss is not None:
            kwargs["loss"] = source_model.compiled_loss._user_losses
            kwargs["loss_weights"] = source_model.compiled_loss._user_loss_weights
        if source_model.compiled_metrics is not None:
            kwargs["metrics"] = source_model.compiled_metrics._user_metrics
            kwargs["weighted_metrics"] = source_model.compiled_metrics._user_weighted_metrics
        target_model.compile(**kwargs)
    return target_model


def patch_compiled_and_attrs(target_obj: object, source_obj: object) -> object:
    """
    Patch compile attributes and other attributes of `source_obj` to `target_obj`.

    :param target_obj: target object
    :param source_obj: source object
    :return: `target_obj`
    """
    patch_compiled(target_obj, source_obj)
    return patch_attrs(target_obj, source_obj)
