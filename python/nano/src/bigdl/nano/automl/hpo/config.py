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

import copy
import warnings
import importlib


from bigdl.nano.automl.utils.register_modules import (
    COMPONENT_TYPE,
    register_module,
    register_module_simple,
    clean_modules_simple,)


TF_LAYER_MODULES = [("tensorflow.keras.layers", "keras.layers"), ]

NANO_DEFINED_TF_LAYERS = ['Embedding']

TF_ACTIVATION_MODULE = [("tensorflow.keras.activations", "keras.activations")]

TF_ACTIVATION_EXCLUDE = ['serialize', 'deserialize', 'get']

TF_FUNCS = ['cast']
TF_KERAS_FUNCS = ['Input']


class HPOConfig(object):
    """A global configuration object for HPO,
    To access it, use "bigdl.nano.automl.hpo_config".
    E.g., "use bigd.nano.automl.hpo_config.enable_hpo_tf() to enable tf hpo"
    """
    def __init__(self,
                 hpo_tf=False,
                 hpo_pytorch=False):
        # configuraitons
        self.hpo_tf_ = hpo_tf
        self.hpo_pytorch_ = hpo_pytorch

        self.torch_available = False
        try:
            import torch
            self.torch_available = True
        except ImportError:
            pass
        self.tf_available = False
        try:
            import tensorflow
            self.tf_available = True
        except ImportError:
            pass

        self.added_tf_activations = []
        self.added_tf_layers = []
        self.backup_tf_layers = None

    def enable_hpo_tf(self):
        if self.hpo_tf:
            return
        self.hpo_tf_ = True if self.tf_available else False
        if self.hpo_tf:
            self._reload_modules()
            self._add_decorated_nano_tf_modules()

    def enable_hpo_pytorch(self):
        self.hpo_pytorch_ = True if self.torch_available else False
        # TODO anything pytorch specific add here

    def disable_hpo_tf(self):
        if self.hpo_tf:
            self._clean_nano_tf_modules()
        # self._reload_modules()
        self.hpo_tf_ = False

    def disable_hpo_pytorch(self):
        self.hpo_pytorch_ = False

    def reset(self):
        self._clean_nano_tf_modules()
        self._reload_modules()

    @property
    def hpo_tf(self):
        return self.hpo_tf_

    @hpo_tf.setter
    def hpo_tf(self, value):
        raise ValueError("Directly set hpo_tf value is not permitted. Please\
            use enable_hpo_tf() or disable_hpo_tf() to enable/disable tensorflow hpo. ")

    @property
    def hpo_pytorch(self):
        return self.hpo_pytorch_

    def _backup_existing_components(self, symtab, subcomponents):
        self.backup_tf_layers = self.backup_tf_layers or {}
        for c in subcomponents:
            self.backup_tf_layers[c] = symtab[c]

    def _restore_existing_components(self, symtab):
        self.backup_tf_layers = self.backup_tf_layers or {}
        for c, component in self.backup_tf_layers.items():
            symtab[c] = component

    def _add_decorated_nano_tf_modules(self):
        # register decorated activations
        import bigdl.nano.tf.keras.activations as nano_activations
        # self.backup_module_symtab(
        #     vars(nano_activations),
        #     self._backup_tf_activations)
        self.added_tf_activations = register_module(
            vars(nano_activations),
            TF_ACTIVATION_MODULE,
            include_types=COMPONENT_TYPE.FUNC,
            exclude_names=TF_ACTIVATION_EXCLUDE)

        # register decorated layers
        import bigdl.nano.tf.keras.layers as nano_layers
        # replace the nano defined layers with decorated layers,
        # TODO auto detect the layers that's been defined in nano.tf.keras
        # instead of using a fixed list NANO_DEFINED_TF_LAYERS
        # register other nano layers defined in keras
        self.added_tf_layers = register_module(
            vars(nano_layers),
            TF_LAYER_MODULES,
            include_types=COMPONENT_TYPE.CLASS,
            exclude_names=NANO_DEFINED_TF_LAYERS)
        self._backup_existing_components(
            vars(nano_layers),
            subcomponents=NANO_DEFINED_TF_LAYERS)
        register_module_simple(
            vars(nano_layers),
            subcomponents=NANO_DEFINED_TF_LAYERS,
            component_type=COMPONENT_TYPE.CLASS,
            module='bigdl.nano.tf.keras.layers'
        )
        self.added_tf_layers.extend(NANO_DEFINED_TF_LAYERS)

        # register decorated tf.cast
        import bigdl.nano.tf
        register_module_simple(vars(bigdl.nano.tf),
                               subcomponents=TF_FUNCS,
                               component_type=COMPONENT_TYPE.FUNC,
                               module='tensorflow')

        # register decorated tf.keras.Input
        import bigdl.nano.tf.keras
        register_module_simple(vars(bigdl.nano.tf.keras),
                               subcomponents=TF_KERAS_FUNCS,
                               component_type=COMPONENT_TYPE.FUNC,
                               module='tensorflow.keras')

    def _reload_modules(self):
        import bigdl.nano.tf.keras.layers
        importlib.reload(bigdl.nano.tf.keras.layers)
        import bigdl.nano.tf.keras.activations
        importlib.reload(bigdl.nano.tf.keras.activations)
        import bigdl.nano.tf.keras
        importlib.reload(bigdl.nano.tf.keras)
        import bigdl.nano.tf
        importlib.reload(bigdl.nano.tf)

    def _clean_nano_tf_modules(self):
        # TODO check all decorated objects and remove them
        # especially for dynamically added layers and activations
        self.added_tf_layers = self.added_tf_layers or []
        self.added_tf_activations = self.added_tf_activations or []
        # clean nano tf layers
        import bigdl.nano.tf.keras.layers as nano_layers
        clean_modules_simple(vars(nano_layers),
                             subcomponents=self.added_tf_layers)
        # restore non-decorated layers in nano, e.g. Embedding
        self._restore_existing_components(vars(nano_layers))

        # clean nano nano_activations
        import bigdl.nano.tf.keras.activations as nano_activations
        clean_modules_simple(vars(nano_activations),
                             subcomponents=self.added_tf_activations)

        # clean up decorated tf.cast
        import bigdl.nano.tf
        clean_modules_simple(vars(bigdl.nano.tf),
                             subcomponents=TF_FUNCS)

        # clean up tf.keras.Input
        import bigdl.nano.tf.keras
        clean_modules_simple(vars(bigdl.nano.tf.keras),
                             subcomponents=TF_KERAS_FUNCS)
