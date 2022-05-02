"""Optimizer wrapper for deferred gradient application."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, List

from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding


class GradientAggregationOptimizer(tf.train.Optimizer):
  """Optimizer wrapper providing deferred gradient application.

  Large hardware configurations are in high-demand, and difficult to come by.
  This class enables simulating execution on large hardware configurations, by
  accumulating gradients across multiple batches. A few caveats apply:

  * Batch statistics (e.g. batch norm) will continue to be based on the
   "micro-batch" size.
  * This effectively trades off computation/time: simulating a large cluster on
    a single core will take an excessive amount of time.

  N.B. Learning rate schedules may need to be adjusted in addition to using
  this optimizer. Schedules should either be scaled down by the relative batch
  size, or use a schedule based on the number of examples to be consistent
  across different batch sizes.
  """

  def __init__(self,
               opt: tf.train.Optimizer,
               grad_steps: int,
               apply_crs_to_grad=False,
               xla_num_partitions=None,
               use_tpu=False):
    self._opt = opt
    self._grad_steps = grad_steps
    self._counter = None
    self._use_tpu = use_tpu
    self._apply_crs_to_grad = apply_crs_to_grad
    self._xla_num_partitions = xla_num_partitions
    self.strategy = tf.distribute.get_strategy()

  def _create_slots(self, var_list):
    """Creates variables for gradient accumulation."""
    if self._use_tpu and not self._counter:
      # only create self counter on tpus to avoid issue with GPU
      self._counter = tf.get_variable(
          shape=[], initializer=tf.zeros_initializer, name='update_count')

    for v in var_list:
      self._opt._zeros_slot(v, 'grad_accum', 'GradientAccumulator')  # pylint: disable=protected-access

  def compute_gradients(self, loss, var_list, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def _sharding(self, x):
    if self._xla_num_partitions:
      # The current partitions are hard coded for Transformer/BERT like
      # models. TODO: make it more general for other models.
      if len(x.get_shape()) == 3:
        x = xla_sharding.split(
            x, 1, self._xla_num_partitions, use_sharding_op=True)
      if len(x.get_shape()) == 2:
        if x.get_shape().as_list()[0] < x.get_shape().as_list()[1]:
          x = xla_sharding.split(
              x, 1, self._xla_num_partitions, use_sharding_op=True)
        else:
          x = xla_sharding.split(
              x, 0, self._xla_num_partitions, use_sharding_op=True)
    return x

  def _apply_and_zero_for_each_replica(self, global_step, accums, var_list):
    normalized_accums = accums
    if self._apply_crs_to_grad:
      normalized_accums = [
          tf.tpu.cross_replica_sum(accum.read_value()) for accum in accums
      ]
    apply_op = self._opt.apply_gradients(
        list(zip(normalized_accums, var_list)))
    with tf.control_dependencies([apply_op]):
      zero_op = [tf.assign(accum, tf.zeros_like(accum)) for accum in accums]
    with tf.control_dependencies([tf.group(zero_op)]):
      # the next line creates a tensor equal to global_step + 1, and doesn't
      # increase global_step itself; used as cross GPU sync only
      return tf.add(global_step, 1)

  def _apply_and_zero(self, distribution, global_step, accums, var_list):
    call_return = distribution.extended.call_for_each_replica(
        self._apply_and_zero_for_each_replica,
        args=(global_step, accums, var_list))
    reduced_call_return = distribution.reduce(
        tf.distribute.ReduceOp.MEAN, call_return, axis=None)
    with tf.control_dependencies([reduced_call_return]):
      return tf.assign_add(global_step, 1)

  def _accum(self, global_step):
    return tf.assign_add(global_step, 1)

  def _maybe_apply_grads_and_zero(self, distribution, global_step, accum_grads,
                                  var_list):
    cond_return = tf.cond(
        tf.equal(tf.mod(global_step, self._grad_steps), self._grad_steps - 1),
        lambda: self._apply_and_zero(
            distribution, global_step, accum_grads, var_list),
        lambda: self._accum(global_step))
    return cond_return

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    grad_list = []
    var_list = []
    for g, v in grads_and_vars:
      grad_list.append(g)
      var_list.append(v)
    with tf.init_scope():
      self._create_slots(var_list)

    # accumulate gradients
    accums = []
    for g, v in zip(grad_list, var_list):
      accum = self.get_slot(v, 'grad_accum')
      # pytype: disable=attribute-error
      if isinstance(g, tf.IndexedSlices):
        scaled_grad = tf.IndexedSlices(g.values / self._grad_steps,
                                       g.indices,
                                       dense_shape=g.dense_shape)
        accums.append(
            accum.assign(self._sharding(accum.read_value()) + scaled_grad))
      else:
        accums.append(
            accum.assign(
                self._sharding(accum.read_value()) + g / self._grad_steps))
      # pytype: enable=attribute-error

    if self._use_tpu:
      def _apply_and_zero_tpu2():
        normalized_accums = accums
        if self._apply_crs_to_grad:
          normalized_accums = [
              tf.tpu.cross_replica_sum(accum.read_value()) for accum in accums
          ]
        apply_op = self._opt.apply_gradients(
            list(zip(normalized_accums, var_list)))
        with tf.control_dependencies([apply_op]):
          zero_op = [tf.assign(accum, tf.zeros_like(accum)) for accum in accums]
        return tf.group(zero_op, tf.assign_add(global_step, 1))

      def _accum_tpu2():
        return tf.group(tf.no_op(), tf.assign_add(global_step, 1))

      accum_step = tf.cond(
          tf.equal(
              tf.mod(self._counter, self._grad_steps), self._grad_steps - 1),
          _apply_and_zero_tpu2, _accum_tpu2)

      with tf.control_dependencies([tf.group(accums)]):
        return tf.group(accum_step, tf.assign_add(self._counter, 1))

    # for GPUs, use merge_call outside tf.cond to avoid issues
    with tf.control_dependencies([tf.group(accums)]):
      merge_return = tf.distribute.get_replica_context().merge_call(
          self._maybe_apply_grads_and_zero,
          args=(global_step, accums, var_list))

    return merge_return

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    return self._opt.variables()

