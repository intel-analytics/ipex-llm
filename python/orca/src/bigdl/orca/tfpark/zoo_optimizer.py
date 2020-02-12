#
# Copyright 2018 Analytics Zoo Authors.
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
import tensorflow as tf
from bigdl.optim.optimizer import OptimMethod
from zoo.util.tf import process_grad


class FakeOptimMethod(OptimMethod):

    def __init__(self):
        super(FakeOptimMethod, self).__init__(None, "float")


class ZooOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    combine gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, name=None):
        if name is None:
            name = "Zoo{}".format(type(optimizer).__name__)
        super(ZooOptimizer, self).__init__(name=name, use_locking=False)

        self._optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.
        See Optimizer.compute_gradients() for more info.
        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        results = []
        for grad_var in gradients:
            grad = grad_var[0]
            var = grad_var[1]
            grad = process_grad(grad)
            with tf.control_dependencies([var]):
                grad_i = tf.identity(grad, name="zoo_identity_op_for_grad")
            results.append((grad_i, var))
        return results

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)

    def _resource_apply_sparse(self, *args, **kwargs):
        self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _resource_apply_dense(self, *args, **kwargs):
        self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _apply_sparse(self, *args, **kwargs):
        self._optimizer._apply_sparse(*args, **kwargs)

    def _apply_dense(self, *args, **kwargs):
        self._optimizer._apply_dense(*args, **kwargs)
