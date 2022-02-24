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
from bigdl.nano.automl.hpo import obj
import inspect


MODULES = [
        "tensorflow.keras.layers",
    ]
EXCLUDE_LAYERS = tuple(['Embedding'])

#INCLUDE_LAYERS = tuple(['Dense'])

def create_layer(module, layer_name):
    layer_class = getattr(module, layer_name)
    derived = type(layer_name, (layer_class,), {})
    decorated = obj()(derived)
    return decorated

def filter_layers(attrs):
    layers = []
    for attr_name, attr in attrs:
        if inspect.isclass(attr):
            module = inspect.getmodule(attr)
            if module.__name__.startswith('keras.layers') \
                and attr.__name__ not in EXCLUDE_LAYERS:
                    layers.append(attr_name)
    return layers

def register_layers(symtab):
    for m in MODULES:
        module = importlib.import_module(m)
        all_layers = filter_layers(vars(module).items())
        # TODO check layers
        for layer_name in all_layers:
            new_cls = create_layer(module, layer_name)
            symtab[layer_name] = new_cls


