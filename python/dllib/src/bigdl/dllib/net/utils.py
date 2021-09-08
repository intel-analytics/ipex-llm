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

import sys
import warnings

if sys.version >= '3':
    long = int
    unicode = str


def find_tensors(sources, predicate):
    '''
    find all the tensors that are used for computing grads and has been
    computed during forward
    :param grads:
    :param forward_ops:
    :return:
    '''
    from collections import deque
    import tensorflow as tf
    queue = deque([])
    for source in sources:
        queue.append(source)

    results = set()
    visited = set()
    while len(queue) > 0:
        node = queue.popleft()
        # this is necessary, because input may not be differentiable
        if node is None:
            continue
        else:
            visited.add(node.name)
            if predicate(node):
                results.add(node)
            else:
                if isinstance(node, tf.Tensor):
                    inputs = list(node.op.inputs) + list(node.op.control_inputs)
                elif isinstance(node, tf.Operation):
                    inputs = list(node.inputs) + list(node.control_inputs)
                else:
                    raise ValueError("Unrecognized Node: {}".format(node))
                for input_tensor in inputs:
                    # this is necessary because there may be a cycle in the graph
                    # such as tf.while_loop
                    if input_tensor.name not in visited:
                        queue.append(input_tensor)
    return list(results)


def find_placeholders(grads):
    import tensorflow as tf

    def predicate(t):
        if not isinstance(t, tf.Operation):
            return t.op.type.startswith("Placeholder")
        else:
            return False

    return find_tensors(grads, predicate)


def _check_the_same(all_required_inputs, inputs_in_datasets):
    inputs_not_in_dataset = [i for i in all_required_inputs if i not in inputs_in_datasets]
    if inputs_not_in_dataset:
        raise ValueError("You should not use any placeholder that are not defined in dataset, " +
                         "found %s" % inputs_not_in_dataset)
    if len(inputs_in_datasets) != len(all_required_inputs):
        inputs_not_require_by_loss = [i for i in inputs_in_datasets if i not in
                                      all_required_inputs]
        raise ValueError("You should use all the placeholders that are defined in dataset, " +
                         "%s are not used" % inputs_not_require_by_loss)


def to_bigdl_optim_method(koptim_method):
    # koptim_method is always an object
    import tensorflow.keras.backend as K
    import tensorflow.keras.optimizers as koptimizers
    import bigdl.optim.optimizer as boptimizer
    import tensorflow.train as tftrain
    import tensorflow as tf
    from tensorflow.python.keras.optimizers import TFOptimizer

    if isinstance(koptim_method, dict):
        res = dict()
        for name, optim_method in koptim_method.items():
            res[name] = to_bigdl_optim_method(optim_method)
        return res

    if isinstance(koptim_method, TFOptimizer):
        koptim_method = koptim_method.optimizer

    if isinstance(koptim_method, boptimizer.OptimMethod):
        return koptim_method
    elif isinstance(koptim_method, koptimizers.Optimizer):
        lr = float(K.eval(koptim_method.lr))
        decay = float(K.eval(koptim_method.decay))
        if isinstance(koptim_method, koptimizers.Adagrad):
            warnings.warn("For Adagrad, we don't support epsilon for now")
            return boptimizer.Adagrad(learningrate=lr,
                                      learningrate_decay=decay)
        elif isinstance(koptim_method, koptimizers.SGD):
            momentum = float(K.eval(koptim_method.momentum))
            return boptimizer.SGD(learningrate=lr,
                                  learningrate_decay=decay,
                                  momentum=momentum,
                                  nesterov=koptim_method.nesterov)
        elif isinstance(koptim_method, koptimizers.Adam):
            beta1 = float(K.eval(koptim_method.beta_1))
            beta2 = float(K.eval(koptim_method.beta_2))
            return boptimizer.Adam(learningrate=lr,
                                   learningrate_decay=decay,
                                   beta1=beta1,
                                   beta2=beta2,
                                   epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.RMSprop):
            rho = float(K.eval(koptim_method.rho))
            return boptimizer.RMSprop(learningrate=lr,
                                      learningrate_decay=decay,
                                      decayrate=rho,
                                      epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.Adadelta):
            warnings.warn(
                "For Adadelta, we don't support learning rate and learning rate decay for now")
            return boptimizer.Adadelta(decayrate=koptim_method.rho,
                                       epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.Adamax):
            beta1 = float(K.eval(koptim_method.beta_1))
            beta2 = float(K.eval(koptim_method.beta_2))
            warnings.warn("For Adamax, we don't support learning rate decay for now")
            return boptimizer.Adamax(learningrate=lr,
                                     beta1=beta1,
                                     beta2=beta2,
                                     epsilon=koptim_method.epsilon)
    elif isinstance(koptim_method, tftrain.Optimizer):
        def get_value(v):
            if isinstance(v, (tf.Tensor, tf.SparseTensor, tf.Variable)):
                return float(K.eval(v))
            else:
                return float(v)

        if isinstance(koptim_method, tftrain.GradientDescentOptimizer):
            lr = get_value(koptim_method._learning_rate)
            return boptimizer.SGD(learningrate=lr)
        elif isinstance(koptim_method, tftrain.MomentumOptimizer):
            lr = get_value(koptim_method._learning_rate)
            momentum = get_value(koptim_method._momentum)
            use_nesterov = koptim_method._use_nesterov
            return boptimizer.SGD(learningrate=lr, momentum=momentum, nesterov=use_nesterov)
        elif isinstance(koptim_method, tftrain.AdagradOptimizer):
            lr = get_value(koptim_method._learning_rate)
            return boptimizer.Adagrad(learningrate=lr)
        elif isinstance(koptim_method, tftrain.AdamOptimizer):
            lr = get_value(koptim_method._lr)
            beta1 = get_value(koptim_method._beta1)
            beta2 = get_value(koptim_method._beta2)
            epsilon = get_value(koptim_method._epsilon)
            return boptimizer.Adam(learningrate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        elif isinstance(koptim_method, tftrain.RMSPropOptimizer):
            lr = get_value(koptim_method._learning_rate)
            decay = get_value(koptim_method._decay)
            momentum = get_value(koptim_method._momentum)
            epsilon = get_value(koptim_method._epsilon)
            centered = get_value(koptim_method._centered)
            if momentum != 0.0 or centered:
                warnings.warn(
                    "For RMSPropOptimizer, we don't support momentum and centered for now")
            return boptimizer.RMSprop(learningrate=lr,
                                      learningrate_decay=decay,
                                      epsilon=epsilon)
        elif isinstance(koptim_method, tftrain.AdadeltaOptimizer):
            lr = get_value(koptim_method._lr)
            rho = get_value(koptim_method._rho)
            epsilon = get_value(koptim_method._epsilon)
            warnings.warn(
                "For Adadelta, we don't support learning rate for now")
            return boptimizer.Adadelta(decayrate=rho, epsilon=epsilon)

    raise ValueError("We don't support %s for now" % koptim_method)
