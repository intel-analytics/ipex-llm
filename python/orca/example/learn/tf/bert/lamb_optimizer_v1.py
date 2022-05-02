"""LAMB (Layer-wise Adaptive Moments) optimizer as TF1 tf.train.Optimizer.

See paper [Large Batch Optimization for Deep Learning: Training BERT in 76
minutes](https://arxiv.org/abs/1904.00962).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
# pylint: enable=g-direct-tensorflow-import


class LAMBOptimizer(optimizer.Optimizer):
  """Optimizer that implements the LAMBOptimizer as tf.train.Optimizer."""

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               weight_decay_rate=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               use_locking=False,
               name="LAMB",
               clip_by_global_norm_after_gradient_allreduce=False):
    super(LAMBOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta_1
    self._beta2 = beta_2
    self._epsilon = epsilon
    self._weight_decay_rate = weight_decay_rate
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.clip_by_global_norm_after_gradient_allreduce = clip_by_global_norm_after_gradient_allreduce
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    self._weight_decay_rate_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)
    weight_decay_rate = self._call_if_callable(self._weight_decay_rate)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
    self._weight_decay_rate_t = ops.convert_to_tensor(
        weight_decay_rate, name="weight_decay_rate")

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables."""
    if not self.clip_by_global_norm_after_gradient_allreduce:
      return super(LAMBOptimizer, self).apply_gradients(
          grads_and_vars, global_step, name)

    tf.logging.info("clip_by_global_norm within LAMB optimizer.")

    grads = []
    vars_ = []
    for g, v in grads_and_vars:
      grads.append(g)
      vars_.append(v)

    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    return super(LAMBOptimizer, self).apply_gradients(
        list(zip(grads, vars_)), global_step, name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(self._weight_decay_rate_t,
                                        var.dtype.base_dtype)
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = m * beta1_t + m_scaled_g_values
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = v * beta2_t + v_scaled_g_values
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

    # ==== The following is with m_t_hat and v_t_hat
    m_t_hat = m_t / (1. - beta1_power)
    v_t_hat = v_t / (1. - beta2_power)

    v_sqrt = math_ops.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + epsilon_t)

    # ==== The following is the original LAMBOptimizer implementation
    # v_sqrt = math_ops.sqrt(v_t_hat)
    # update = m_t / (v_sqrt + epsilon_t)

    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
      update += weight_decay_rate_t * var

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(
          math_ops.greater(w_norm, 0),
          array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0),
          1.0)

    var_update = var - ratio * lr_t * update
    return state_ops.assign(var, var_update, use_locking=self._use_locking).op

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(self._weight_decay_rate_t,
                                        var.dtype.base_dtype)
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    # ==== The following is with m_t_hat and v_t_hat
    m_t_hat = m_t / (1. - beta1_power)
    v_t_hat = v_t / (1. - beta2_power)

    v_sqrt = math_ops.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + epsilon_t)

    # ==== The following is the original LAMBOptimizer implementation
    # v_sqrt = math_ops.sqrt(v_t_hat)
    # update = m_t / (v_sqrt + epsilon_t)

    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
      update += weight_decay_rate_t * var

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(
          math_ops.greater(w_norm, 0),
          array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0),
          1.0)
    var_update = state_ops.assign_sub(
        var, ratio * lr_t * update, use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
