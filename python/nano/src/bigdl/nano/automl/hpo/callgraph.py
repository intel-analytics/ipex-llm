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

import networkx as nx
from numpy import isin
import tensorflow as tf
from .space import AutoObject
from .backend import OptunaBackend
from tensorflow.python.training.tracking.data_structures import ListWrapper

def create_callgraph():
    call_g = nx.DiGraph()
    return call_g


def update_callgraph(inputs, current):
    # TODO: handle multiple input case
    call_g = create_callgraph()
    if not isinstance(inputs, list):
        inputs = [inputs]
    for inp in inputs:
        if tf.is_tensor(inp):
            #call_g = create_callgraph()
            # Tensor is not hashable so add ref instead
            call_g.add_node(inp.ref(), out_tensor=inp)
            call_g.add_edge(inp.ref(), current)
        elif isinstance(inp, AutoObject):
            assert(inp._callgraph is not None)
            # merge call graph from the input
            call_g.update(inp._callgraph)
            call_g.add_edge(inp, current)
    return call_g


def exec_callgraph(inputs, outputs, trial):
    """ inputs - input Tensor
        outputs - auto objects
        return - output Tensor
    """
    # TODO: need topological sort?
    g = outputs._callgraph
    # if isinstance(inputs, ListWrapper):
    #     source = [i.ref() for i in inputs]
    # elif tf.is_tensor(inputs):
    #     source = inputs.ref()
    # else:
    #     raise ValueError("Unrecognized input type:" + type(inputs))
    #traverse_list = nx.dfs_predecessors(g, source=source).items()
    predecessors = nx.dfs_predecessors(g)
    sorted_nodes = list(nx.topological_sort(g))

    #for autolayer, parent in sorted_nodes:
    for node in sorted_nodes:
        if not isinstance(node, AutoObject):
            continue
        parents = predecessors[node]
        if isinstance(parents, list):
            in_tensors = [g.nodes[p]['out_tensor'] for p in parents]
        else:
            in_tensors = g.nodes[parents]['out_tensor']
        # layer is an auto object
        layer = OptunaBackend.instantiate(trial, node)
        out_tensor = layer(in_tensors)
        g.add_node(node, out_tensor=out_tensor)
    outs = g.nodes[outputs]['out_tensor']
    return outs


def plot_callgraph(g, save_path="callgraph_plot.png"):
    #nx.draw(g, pos=nx.shell_layout(g))
    nx.draw_networkx(g,
                     pos=nx.spring_layout(g),
                     font_size=5,
                     node_size=30)
    import matplotlib.pyplot as plt
    plt.savefig(save_path)
