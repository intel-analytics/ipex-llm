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
from bigdl.nano.automl.hpo import obj, func
from bigdl.nano.utils.log4Error import invalidInputError
import inspect
from enum import Enum
import copy


class COMPONENT_TYPE(Enum):
    CLASS = 1
    FUNC = 2


def decorate_cls(module, name):
    component = getattr(module, name)
    derived = type(name, (component,), {})
    decorated = obj()(derived)
    return decorated


def decorate_func(module, name):
    component = getattr(module, name)
    decorated = func()(component)
    return decorated


def register_module(target_symtab,
                    modules,
                    include_types,
                    exclude_names):
    '''register subcomponents in modules into target symtable,
    return added components for later clean up'''
    # get a component and return the auto counterpart

    # filter the classes in a module
    def filter(module,
               prefix,
               check_type,
               exclude_set):
        filtered = []
        # attrs = vars(module).items()
        # for name, attr in attrs:
        #     if check_type(attr):
        #         m = inspect.getmodule(attr)
        #         if m.__name__.startswith(prefix) and name not in exclude_set:
        #             filtered.append(name)

        attrs = dir(module)
        for name in attrs:
            attr = getattr(module, name)
            if not name.startswith('_') and check_type(attr):
                m = inspect.getmodule(attr)
                if m.__name__.startswith(prefix) and name not in exclude_set:
                    filtered.append(name)
        return filtered

    if include_types == COMPONENT_TYPE.CLASS:
        def type_checker(x):
            return inspect.isclass(x)
        decorator = decorate_cls
    elif include_types == COMPONENT_TYPE.FUNC:
        def type_checker(x):
            return inspect.isfunction(x)
        decorator = decorate_func
    else:
        invalidInputError(False,
                          "Unknown Component Type, should be either class or function")

    added_components = []
    for m, prefix in modules:
        module = importlib.import_module(m)
        c_names = filter(module,
                         prefix,
                         check_type=type_checker,
                         exclude_set=exclude_names)
        # TODO check layers
        for c_name in c_names:
            new_c = decorator(module, c_name)
            if target_symtab.get(c_name, None) is not None:
                invalidInputError(False,
                                  "Fail to register decorated component to the target nano"
                                  " module, as it is already defined in the target module."
                                  "Use register_module_simple instead.")
            target_symtab[c_name] = new_c
            added_components.append(c_name)

    return added_components


def register_module_simple(target_symtab,
                           subcomponents,
                           component_type,
                           module):
    '''register subcomponents in a module to target symtable,
    a simple version without many checks as in register_module.
    subcomponents must be within the vars() of the module'''
    if component_type == COMPONENT_TYPE.CLASS:
        decorator = decorate_cls
    elif component_type == COMPONENT_TYPE.FUNC:
        decorator = decorate_func
    m = importlib.import_module(module)
    for c in subcomponents:
        new_f = decorator(m, c)
        target_symtab[c] = new_f


def clean_modules_simple(target_symtab,
                         subcomponents):
    '''clean up the subcomponents in target symtab'''
    for f in subcomponents:
        target_symtab.pop(f, None)
