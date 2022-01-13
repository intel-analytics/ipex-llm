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
# ==============================================================================

import numpy as np
from sparse_adam import SparseAdam

def test_optimizer_sparseadam():

    from bigdl.nano.tf.keras.layers import Embedding
    from bigdl.nano.tf.keras import Sequential
    import tensorflow as tf
    model = Sequential()
    model.add(Embedding(1000, 64, input_length=10, regularizer=tf.keras.regularizers.L2()))
    sparse_optim = SparseAdam()
    model.compile(sparse_optim, 'mse')
    input_array = np.random.randint(1000, size=(32, 10))

    ids = tf.constant([0, 10, 20, 30])
    test_array = tf.nn.embedding_lookup(input_array,ids)
    print(test_array)
    with tf.GradientTape() as tape:
        output = model(test_array)
        labels = tf.zeros_like(output)
        loss_value = model.compiled_loss(output, labels, regularization_losses=model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        for grad in grads:
            print("grad:",grad)
            assert isinstance(grad, tf.IndexedSlices)