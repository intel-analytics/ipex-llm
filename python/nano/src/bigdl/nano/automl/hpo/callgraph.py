import networkx as nx
import tensorflow as tf
from .space import AutoObject
from .backend import OptunaBackend

def create_callgraph():
    call_g = nx.DiGraph()
    return call_g

def update_callgraph(inputs, current):
    #TODO: handle multiple input case
    if tf.is_tensor(inputs):
        call_g = create_callgraph()
        # Tensor is not hashable so add ref instead
        call_g.add_node(inputs.ref(), tensor=inputs)
        call_g.add_edge(inputs.ref(),current)
    elif isinstance(inputs, AutoObject):
        assert(inputs._callgraph is not None)
        call_g = inputs._callgraph
        call_g.add_edge(inputs,current)
    return call_g


def exec_callgraph(inputs, outputs, trial):
    """ inputs - input Tensor
        outputs - auto objects
        return - output Tensor
    """
    # TODO: need topological sort?
    g = outputs._callgraph
    traverse_list = nx.dfs_predecessors(g, source=inputs.ref()).items()
    for autolayer, parent in traverse_list:
        #TODO handle parent is a list
        in_tensor = g.nodes[parent]['tensor']
        # layer is an auto object
        assert(isinstance(autolayer, AutoObject))
        layer=OptunaBackend.instantiate(trial, autolayer)
        out_tensor = layer(in_tensor)
        g.add_node(autolayer, tensor=out_tensor)
    outs = g.nodes[outputs]['tensor']
    return outs

def plot_callgraph(g):
    nx.draw(g, with_labels=True, font_weight='bold')
