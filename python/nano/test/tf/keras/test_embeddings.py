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


def test_embedding_gradient_sparse():

    import bigdl.nano.tf.keras.layers.Embeddings
    import bigdl.nano.tf.keras.Sequential
    import tensorflow as tf
    model = Sequential()
    model.add(Embedding(1000, 64, input_length=10, regularizer=tf.keras.regularizer.L2()))
    model.compile('rmsprop', 'mse')
    input_array = np.random.randint(1000, size=(32, 10))

    with tf.GradientTape() as tape:
        output = model(input_array)
        labels = tf.zeros_like(output)
        loss_value = model.compiled_loss(output, labels, regularization_losses=model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)

        for grad in grads:
            assert isinstance(grad, tf.IndexedSlices)