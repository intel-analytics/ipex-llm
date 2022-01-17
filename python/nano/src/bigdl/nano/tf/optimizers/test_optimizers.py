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
    model.add(Embedding(30, 16, input_length=5))

    sparse_optim = SparseAdam()
    model.compile(sparse_optim, 'mse')

    input_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    print("before_weight:", model.get_weights())
    before_weights = model.get_weights()

    output = model(input_array)
    labels = np.random.randint(30, size=output.shape)

    y_pred = model.train_on_batch(input_array, labels)

    print("after_weight:", model.get_weights())
    after_weights = model.get_weights()

    assert (before_weights[0][0:10, :].all() != after_weights[0][0:10, :].all())
