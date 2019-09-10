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

if sys.version >= '3':
    long = int
    unicode = str


def _find_placeholders(grads):
    '''
    find all the tensors that are used for computing grads and has been
    computed during forward
    :param grads:
    :param forward_ops:
    :return:
    '''
    from collections import deque
    queue = deque([])
    for grad in grads:
        queue.append(grad)

    placeholders = set()
    visited = set()
    while len(queue) > 0:
        tensor = queue.popleft()
        # this is necessary, because input may not be differentiable
        if tensor is None:
            continue
        else:
            visited.add(tensor.name)
            if tensor.op.type.startswith("Placeholder"):
                placeholders.add(tensor)
                continue
            for input_tensor in tensor.op.inputs:
                # this is necessary because there may be a cycle in the graph such as tf.while_loop
                if input_tensor.name not in visited:
                    queue.append(input_tensor)
    return list(placeholders)


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
