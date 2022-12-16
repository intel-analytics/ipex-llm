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


def mixedprecision_train_step_generator(model, loss, optimizer):
    '''
    Running a customized training loop with mixed precision requires changes over
    running it in float32, especially need to use loss scaling if mixed_float16 is used.

    :param model: The model to train.
    :param loss: tf.keras loss instance.
    :param optimizer: tf.keras optimizer instance.

    :return: A training loop function with mixed precision.

    Example:
        >>> # to set global dtype policy
        >>> from bigdl.nano.tf import patch_tensorflow
        >>> patch_tensorflow(precision='mixed_bfloat16')
        >>> # to generate a training loop with mixed precision
        >>> from bigdl.nano.tf.keras import mixedprecision_train_step_generator
        >>> fit = mixedprecision_train_step_generator(model, loss, optimizer)
        >>> # start training
        >>> fit(x, y)
    '''

    from tensorflow.keras import mixed_precision
    import tensorflow as tf
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    @tf.function
    def train_step(x, y):
        x = tf.cast(x, tf.bfloat16)
        y = tf.cast(y, tf.bfloat16)
        with tf.GradientTape() as tape:
            pred = model(x)
            loss_value = loss(y, pred)
            scaled_loss = optimizer.get_scaled_loss(loss_value)
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value
    return train_step
