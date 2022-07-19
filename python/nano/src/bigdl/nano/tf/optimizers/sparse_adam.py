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
"""SparseAdam optimizer implementation."""

import tensorflow.compat.v2 as tf
from keras import backend_config

from tensorflow.python.util.tf_export import keras_export
import tensorflow
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.tf.utils import KERAS_VERSION_LESS_2_9

if KERAS_VERSION_LESS_2_9:
    from keras.optimizer_v2 import optimizer_v2
else:
    from keras.optimizers.optimizer_v2 import optimizer_v2


class SparseAdam(tensorflow.keras.optimizers.Adam):
    """
    A variant of the Adam optimizer that handles sparse updates more efficiently.

    The original Adam algorithm maintains two moving-average accumulators for
    each trainable variable; the accumulators are updated at every step.
    In this variant, only moments that show up in the gradient get updated,
    and only those portions of the gradient get applied to the parameters.
    Compared with the original Adam optimizer, it can provide large improvements in
    model training throughput for some applications.
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='SparseAdam',
                 **kwargs):
        """
        Create a slightly modified version of tf.keras.optimizers.Adam.

        which only update moving-average accumulators for sparse variable
        indices that appear in the current batch.

        :param learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use, The
            learning rate. Defaults to 0.001.
        :param beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        :param beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use, The
            exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
        :param amsgrad: Boolean. Currently amsgrad is not supported and it can only
            set to False.
        :param name: Optional name for the operations created when applying gradients.
            Defaults to `"Adam"`.
        :param kwargs: Keyword arguments. Allowed to be one of
            `"clipnorm"` or `"clipvalue"`.
            `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
            gradients by value.
        """
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        temp_coeff = (apply_state or {}).get((var_device, var_dtype))
        coefficients = (temp_coeff or self._fallback_apply_state(var_device, var_dtype))

        import tensorflow as tf

        with tf.name_scope("update_sparse_mt"):
            m = self.get_slot(var, 'm')
            m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
            m_sparse = tf.gather(m, indices=indices)

            m_sparse_scaled = m_sparse * (coefficients['beta_1_t'] - 1)

            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values + m_sparse_scaled)
            m_t_sparse = m_sparse * coefficients['beta_1_t'] + m_scaled_g_values

        with tf.name_scope("update_sparse_vt"):
            v = self.get_slot(var, 'v')
            v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
            v_sparse = tf.gather(v, indices=indices)
            v_sparse_scaled = v_sparse * (coefficients['beta_2_t'] - 1)

            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values + v_sparse_scaled)

            v_t_sparse = v_sparse * coefficients['beta_2_t'] + v_scaled_g_values

        if not self.amsgrad:
            with tf.name_scope("update_sparse_var"):
                v_sqrt_sparse = tf.sqrt(v_t_sparse)
                var_update = self._resource_scatter_add(
                    var,
                    indices,
                    -1 * coefficients['lr'] * m_t_sparse / (v_sqrt_sparse + coefficients['epsilon'])
                )
                return tf.group(*[var_update, m_t, v_t])
        else:
            invalidInputError(False, "do not support amsgrad for now")
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])
