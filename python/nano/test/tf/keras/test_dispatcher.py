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

from unittest import TestCase


class TestChannelsLast(TestCase):

    def test_dispatch_keras(self):
        from bigdl.nano import patch_nano, unpatch_nano
        patch_nano()
        import keras
        import tensorflow
        import bigdl.nano
        # not checking keras.Model since there is no change
        assert issubclass(tensorflow.keras.Sequential, bigdl.nano.tf.keras.Sequential)
        assert issubclass(keras.Sequential, bigdl.nano.tf.keras.Sequential)
        assert issubclass(keras.layers.Embedding, bigdl.nano.tf.keras.layers.Embedding)
        assert issubclass(tensorflow.keras.layers.Embedding, bigdl.nano.tf.keras.layers.Embedding)
        assert issubclass(tensorflow.keras.optimizers.Adam, bigdl.nano.tf.optimizers.SparseAdam)

        unpatch_nano()
        # not checking keras.Model since there is no change
        assert not issubclass(tensorflow.keras.Sequential, bigdl.nano.tf.keras.Sequential)
        assert not issubclass(keras.Sequential, bigdl.nano.tf.keras.Sequential)
        assert not issubclass(keras.layers.Embedding, bigdl.nano.tf.keras.layers.Embedding)
        assert not issubclass(tensorflow.keras.layers.Embedding, bigdl.nano.tf.keras.layers.Embedding)
        assert not issubclass(tensorflow.optimizers.Adam, bigdl.nano.tf.optimizers.SparseAdam)
