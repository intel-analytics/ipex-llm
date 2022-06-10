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


import pytest
from unittest import TestCase

import bigdl.nano.automl as nano_automl
import bigdl.nano.tf

class TestGlobalConfig(TestCase):

    def _import_should_raise(self):
        with self.assertRaises(ImportError):
            from bigdl.nano.tf.keras.activations import sigmoid, linear
        with self.assertRaises(ImportError):
            from bigdl.nano.tf.keras.layers import Dense, ELU, Embeddings
        with self.assertRaises(ImportError):
            from bigdl.nano.tf import cast
        with self.assertRaises(ImportError):
            from bigdl.nano.tf.keras import Input
        with self.assertRaises(ImportError):
            from bigdl.nano.tf.optimizers import SGD, RMSprop, Adam, SparseAdam

    def _import_should_okay(self):
        try:
            from bigdl.nano.tf.keras.activations import sigmoid, linear
        except ImportError:
            self.fail("nano.tf.keras.activations did not register correctly.")
        try:
            from bigdl.nano.tf.keras.layers import Dense, Embedding
        except ImportError:
            self.fail("nano.tf.keras.layers did not register correctly.")
        try:
            from bigdl.nano.tf import cast
        except ImportError:
            self.fail("bigdl.nano.tf.cast did not register correctly.")
        try:
            from bigdl.nano.tf.keras import Input
        except ImportError:
            self.fail("nano.tf.keras.Input did not register correctly.")
        try:
            from bigdl.nano.tf.optimizers import SGD, RMSprop, Adam, SparseAdam
        except ImportError:
            self.fail("nano.tf.optimizers did not register correctly.")

    def test_enable_automl(self):
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()

    def test_disable_automl(self):
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()

    def test_multi_enable_disable(self):
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()

    def test_multi_enable_disable2(self):
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()
        nano_automl.hpo_config.disable_hpo_tf()
        self._import_should_raise()
        nano_automl.hpo_config.enable_hpo_tf()
        self._import_should_okay()

    def test_hpo_settings(self):
        with self.assertRaises(RuntimeError):
            nano_automl.hpo_config.hpo_tf = True


if __name__ == '__main__':
    pytest.main([__file__])