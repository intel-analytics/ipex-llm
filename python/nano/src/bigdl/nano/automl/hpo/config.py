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


from bigdl.nano.automl.utils.register_modules import (
    COMPONENT_TYPE,
    register_module,
    register_module_simple,
    clean_modules_simple,)
from bigdl.nano.utils.log4Error import invalidInputError


class HPOConfig(object):
    """
    A global configuration object for HPO.

    To access the hpo config, use "bigdl.nano.automl.hpo_config".
    E.g., to enable hpo tf, use "bigd.nano.automl.hpo_config.enable_hpo_tf()"
    """

    # tf.keras.layers.xxx
    TF_LAYER_MODULES = [("tensorflow.keras.layers", "keras.layers"), ]
    NANO_DEFINED_TF_LAYERS = ['Embedding']
    # tf.keras.activations.xxx
    TF_ACTIVATION_MODULE = [("tensorflow.keras.activations", "keras.activations")]
    TF_ACTIVATION_EXCLUDE = ['serialize', 'deserialize', 'get']
    TF_FUNCS = ['cast']  # tf.xxx
    TF_KERAS_FUNCS = ['Input']  # tf.keras.xxx
    # tf.keras.optimizers.xxx
    TF_OPTIMIZERS = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
    NANO_DEFINED_TF_OPTIMIZERS = ['SparseAdam']

    def __init__(self,
                 hpo_tf=False,
                 hpo_pytorch=False):
        """
        Init the HPOConfig.

        :param hpo_tf: boolean. whether to enable HPO for Tensorflow.
            Defaults to False
        :param hpo_pytorch: boolean. whether to enable HPO for PyTorch.
            Defaults to False
        """
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
        self.backup_tf_components = None

    def enable_hpo_tf(self):
        """Enable HPO for tensorflow."""
        if self.hpo_tf:
            return
        self.hpo_tf_ = True if self.tf_available else False
        if self.hpo_tf:
            self._reload_modules()
            self._add_decorated_nano_tf_modules()

    def enable_hpo_pytorch(self):
        """Enable HPO for pytorch."""
        if self.hpo_pytorch:
            return
        self.hpo_pytorch_ = True if self.torch_available else False
        # TODO anything pytorch specific add here

    def disable_hpo_tf(self):
        """Disable HPO for tensorflow."""
        if self.hpo_tf:
            self._clean_nano_tf_modules()
        # self._reload_modules()
        self.hpo_tf_ = False

    def disable_hpo_pytorch(self):
        """Disable HPO for pytorch."""
        self.hpo_pytorch_ = False

    def reset(self):
        """Reset the HPO Config to default values."""
        self._clean_nano_tf_modules()
        self._reload_modules()
        self.hpo_tf_ = False
        self.hpo_pytorch_ = False

    @property
    def hpo_tf(self):
        """Get the status of hpo tensorflow."""
        return self.hpo_tf_

    @hpo_tf.setter
    def hpo_tf(self, value):
        """Forbid setting hpo tensorflow variable directly. \
        Should use enable_hpo_tf() instead."""
        invalidInputError(False, "Directly set hpo_tf value is not permitted."
                                 "Please use enable_hpo_tf() or disable_hpo_tf() to "
                                 "enable/disable tensorflow hpo. ")

    @property
    def hpo_pytorch(self):
        """Get the status of hpo pytorch."""
        return self.hpo_pytorch_

    @hpo_pytorch.setter
    def hpo_pytorch(self, value):
        """Forbid setting hpo pytorch variable directly. \
            Should use enable_hpo_pytorch() instead."""
        invalidInputError(False, "Directly set hpo_pytorch value is not permitted."
                                 "Please use enable_hpo_pytorch() or disable_hpo_pytorch()"
                                 " to enable/disable pytorch hpo. ")

    def _backup_existing_components(self, symtab, subcomponents, namespace='default'):
        self.backup_tf_components = self.backup_tf_components or {}
        for c in subcomponents:
            self.backup_tf_components[(c, namespace)] = symtab[c]

    def _restore_existing_components(self, symtab, namespace='default'):
        self.backup_tf_components = self.backup_tf_components or {}
        for (c, ns), component in self.backup_tf_components.items():
            if ns == namespace:
                symtab[c] = component

    def _add_decorated_nano_tf_modules(self):
        # register decorated activations
        import bigdl.nano.tf.keras.activations as nano_activations
        # self.backup_module_symtab(
        #     vars(nano_activations),
        #     self._backup_tf_activations)
        self.added_tf_activations = register_module(
            vars(nano_activations),
            HPOConfig.TF_ACTIVATION_MODULE,
            include_types=COMPONENT_TYPE.FUNC,
            exclude_names=HPOConfig.TF_ACTIVATION_EXCLUDE)

        # register decorated layers
        import bigdl.nano.tf.keras.layers as nano_layers
        # register tf.keras.layers except nano defined layers
        self.added_tf_layers = register_module(
            vars(nano_layers),
            HPOConfig.TF_LAYER_MODULES,
            include_types=COMPONENT_TYPE.CLASS,
            exclude_names=HPOConfig.NANO_DEFINED_TF_LAYERS)

        # backup the original nano defined layers and replace
        # them with decorated layers,
        # TODO auto detect the layers that's been defined in nano.tf.keras
        # instead of using a fixed list NANO_DEFINED_TF_LAYERS
        self._backup_existing_components(
            vars(nano_layers),
            subcomponents=HPOConfig.NANO_DEFINED_TF_LAYERS,
            namespace='layers')
        register_module_simple(
            vars(nano_layers),
            subcomponents=HPOConfig.NANO_DEFINED_TF_LAYERS,
            component_type=COMPONENT_TYPE.CLASS,
            module='bigdl.nano.tf.keras.layers'
        )
        self.added_tf_layers.extend(HPOConfig.NANO_DEFINED_TF_LAYERS)

        # register decorated tf.cast
        import bigdl.nano.tf
        register_module_simple(vars(bigdl.nano.tf),
                               subcomponents=HPOConfig.TF_FUNCS,
                               component_type=COMPONENT_TYPE.FUNC,
                               module='tensorflow')

        # register decorated tf.keras.Input
        import bigdl.nano.tf.keras
        register_module_simple(vars(bigdl.nano.tf.keras),
                               subcomponents=HPOConfig.TF_KERAS_FUNCS,
                               component_type=COMPONENT_TYPE.FUNC,
                               module='tensorflow.keras')

        # register decoratred tf.keras.optimizers.*
        import bigdl.nano.tf.optimizers as nano_optimizers
        register_module_simple(vars(nano_optimizers),
                               subcomponents=HPOConfig.TF_OPTIMIZERS,
                               component_type=COMPONENT_TYPE.CLASS,
                               module='tensorflow.keras.optimizers')

        # backup nano defined optimizer and replace them
        self._backup_existing_components(
            vars(nano_optimizers),
            subcomponents=HPOConfig.NANO_DEFINED_TF_OPTIMIZERS,
            namespace='optimizers')
        register_module_simple(
            vars(nano_optimizers),
            subcomponents=HPOConfig.NANO_DEFINED_TF_OPTIMIZERS,
            component_type=COMPONENT_TYPE.CLASS,
            module='bigdl.nano.tf.optimizers'
        )

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
        self._restore_existing_components(vars(nano_layers),
                                          namespace='layers')

        # clean nano nano_activations
        import bigdl.nano.tf.keras.activations
        clean_modules_simple(vars(bigdl.nano.tf.keras.activations),
                             subcomponents=self.added_tf_activations)

        # clean up decorated tf.cast
        import bigdl.nano.tf
        clean_modules_simple(vars(bigdl.nano.tf),
                             subcomponents=HPOConfig.TF_FUNCS)

        # clean up tf.keras.Input
        import bigdl.nano.tf.keras
        clean_modules_simple(vars(bigdl.nano.tf.keras),
                             subcomponents=HPOConfig.TF_KERAS_FUNCS)

        # clean up tf.keras.Input
        import bigdl.nano.tf.keras
        clean_modules_simple(vars(bigdl.nano.tf.keras),
                             subcomponents=HPOConfig.TF_KERAS_FUNCS)

        # clean up optimizers
        import bigdl.nano.tf.optimizers as nano_optimizers
        all_optimzers = HPOConfig.TF_OPTIMIZERS + HPOConfig.NANO_DEFINED_TF_OPTIMIZERS
        clean_modules_simple(vars(nano_optimizers),
                             subcomponents=all_optimzers)
        # restore non-decorated layers in nano, e.g. SparseAdam
        self._restore_existing_components(vars(nano_optimizers),
                                          namespace='optimizers')
