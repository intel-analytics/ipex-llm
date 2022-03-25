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
from bigdl.nano.automl.hpo import obj,func
import inspect
from enum import Enum

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

def register_module(symtab,
                    modules,
                    prefix,
                    include_types,
                    exclude_names):
    # get a component and return the auto counterpart

    # filter the excluded modules
    def filter(module_name,
               module,
               check_type,
               exclude_set):
        filtered = []
        attrs= vars(module).items()
        for name, attr in attrs:
            if check_type(attr):
                m = inspect.getmodule(attr)
                if m.__name__.startswith(prefix) \
                    and name not in exclude_set:
                        filtered.append(name)
        return filtered

    if include_types == COMPONENT_TYPE.CLASS:
        type_checker = lambda x: inspect.isclass(x)
        decorator = decorate_cls
    elif include_types == COMPONENT_TYPE.FUNC:
        type_checker = lambda x: inspect.isfunction(x)
        decorator = decorate_func
    else:
        raise ValueError("Unknown Component Type",
                         "should be either class or function")

    for m in modules:
        module = importlib.import_module(m)
        c_names = filter(m,
                         module,
                         check_type = type_checker,
                         exclude_set=exclude_names)
        # TODO check layers
        for c_name in c_names:
            new_c = decorator(module, c_name)
            symtab[c_name] = new_c



def register_layers(symtab):
    layer_modules = [
            "tensorflow.keras.layers",
        ]
    exclude_layers = ['Embedding']
    register_module(symtab,
                    layer_modules,
                    prefix="keras.layers",
                    include_types=COMPONENT_TYPE.CLASS,
                    exclude_names=exclude_layers)

def register_activations(symtab):
    # for activations
    activation_module = [
            "tensorflow.keras.activations"
        ]
    excl_activations= ['serialize','deserialize','get']
    register_module(symtab,
                activation_module,
                prefix="keras.activations",
                include_types=COMPONENT_TYPE.FUNC,
                exclude_names=excl_activations)


def register_tf_funcs(symtab):
    components = [
        'cast',
        ]
    for f in components:
        tf_module = importlib.import_module('tensorflow')
        new_f = decorate_func(tf_module, f)
        symtab[f] = new_f

