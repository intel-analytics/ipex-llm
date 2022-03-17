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
import tensorflow as tf
from .space import AutoObject
from .backend import OptunaBackend
from enum import Enum


class CALLTYPE(Enum):
    LAYER_CALL = 1 # e.g. all keras layers such as Conv2D, etc.
    FUNC_CALL = 2 # such as keras.Input
    FUNC_SLICE = 3 # slices on output of a func, e.g. keras.Input(...)[:,:,]


class CallGraph():

    def __init__(self):
        self.callg_ = nx.DiGraph()

    def update_edges(self, other):
        self.call_g.update(other.call_g)

    def add_edge(self, curr, parent, type=CALLTYPE.LAYER_CALL):
        self.call_g.add_edge(parent, curr)

    def add_tensor(self, node, output):
        self.call_g.add_node(node, out_tensor=output)

    @property
    def calls(self):
        return self.callg_

    @staticmethod
    def create():
        call_g = CallGraph()
        return call_g

    @staticmethod
    def update(inputs, current, type=CALLTYPE.LAYER_CALL):
        call_g = CallGraph.create()
        if not isinstance(inputs, list):
            inputs = [inputs]
        for inp in inputs:
            if tf.is_tensor(inp):
                #call_g = create_callgraph()
                # Tensor is not hashable so add ref instead
                call_g.add_tensor(inp.ref(), inp)
                call_g.add_edge(current,inp.ref())
            elif isinstance(inp, AutoObject):
                assert(inp._callgraph is not None)
                # merge call graph from the input
                call_g.update_edges(inp._callgraph)
                call_g.add_edge(current, inp)
        return call_g


    @staticmethod
    def execute(inputs, outputs, trial):
        """ inputs - input Tensor
            outputs - auto objects
            return - output Tensor
        """
        # TODO: need topological sort?
        g = outputs._callgraph.calls()
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

    @staticmethod
    def plot(g: nx.DiGraph, save_path="callgraph_plot.png"):
        #nx.draw(g, pos=nx.shell_layout(g))
        # nx.draw_networkx(g,
        #                 pos=nx.spring_layout(g),
        #                 font_size=5,
        #                 node_size=30)
        # import matplotlib.pyplot as plt
        # plt.savefig(save_path)
        predecessors = nx.dfs_predecessors(g)
        sorted_nodes = list(nx.topological_sort(g))
        for node in sorted_nodes:
            if not isinstance(node, AutoObject):
                continue
            parents = predecessors[node]
            print(node, parents)



class CallCache():

    def __init__(self):
        self.callqueue_ = []
        self.tensors_ = dict()
        self.skip = False

    def update_tensors(self, other):
        self.tensors_.update(other.tensors_)

    def update_calls(self, other):
        self.callqueue_ += other.callqueue_

    def append_call(self, caller, arguments, type):
        self.callqueue_.append((type, caller, arguments))

    def add_tensor(self, node, tensor):
        self.tensors_[node] = tensor

    def get_tensor(self, n):
        if isinstance(n, list):
            return [self.tensors_.get(one_n, None) for one_n in n]
        else:
            return self.tensors_.get(n, None)

    @property
    def calls(self):
        return self.callqueue_

    @staticmethod
    def create():
        cache = CallCache()
        return cache

    @staticmethod
    def update(arguments, current, type=CALLTYPE.LAYER_CALL):
        def update_cache_from_input(cache, inp):
            if tf.is_tensor(inp):
                return
                #cache.tensors[inp.ref()] = inp
            elif isinstance(inp, AutoObject):
                assert(inp._callgraph is not None)
                input_callgraph = inp._callgraph
                # merge call graph from the input
                if not input_callgraph.skip:
                    cache.update_tensors(input_callgraph)
                    cache.update_calls(input_callgraph)
                    input_callgraph.skip = True

        #print((current, arguments))
        cur_cache = CallCache.create()

        if type == CALLTYPE.LAYER_CALL:
            if isinstance(arguments, list):
                for inp in arguments:
                    update_cache_from_input(cur_cache,inp)
            else:
                update_cache_from_input(cur_cache,arguments)
            cur_cache.append_call(current, arguments, CALLTYPE.LAYER_CALL)
        elif type == CALLTYPE.FUNC_SLICE:
            (source, slice_args) = arguments
            update_cache_from_input(cur_cache, source)
            cur_cache.append_call(current,
                                  arguments,
                                  CALLTYPE.FUNC_SLICE)
        elif type == CALLTYPE.FUNC_CALL:
            cur_cache.append_call(current, arguments, CALLTYPE.FUNC_CALL)
        else:
            raise ValueError("Unexpected CallType: %s" % type)

        return cur_cache


    @staticmethod
    def execute(inputs, outputs, trial):
        def process_node(n, cache):
            if tf.is_tensor(n):
                tensor = n
            else:
                tensor = cache.get_tensor(n)
            return tensor

        out_cache = outputs._callgraph
        #for autolayer, parent in sorted_nodes:

        # get output tensors
        for call_type, caller, arguments in out_cache.calls:
            if call_type == CALLTYPE.LAYER_CALL:
                inp = arguments
                if isinstance(arguments, list):
                    in_tensors = [ process_node(p, out_cache) for p in inp]
                else:
                    in_tensors = process_node(inp, out_cache)
                # layer is an auto object
                if isinstance(caller, AutoObject):
                    instance = OptunaBackend.instantiate(trial, caller)
                else:
                    instance = caller
                # the actual excution of the functional API
                out_tensor = instance(in_tensors)
            elif call_type == CALLTYPE.FUNC_SLICE:
                source, slice_args = arguments
                slice_args, slice_kwargs =slice_args
                source_tensor = out_cache.get_tensor(source)
                # the actual excution of the functional API
                out_tensor = source_tensor.__getitem__(*slice_args, **slice_kwargs)
            elif call_type == CALLTYPE.FUNC_CALL:
                # we disable search spaces in func arguments for now
                out_tensor = OptunaBackend.instantiate(trial, caller)
            else:
                raise ValueError("Unexpected CallType: %s" % type)
            out_cache.add_tensor(caller, out_tensor)
        out_tensors = out_cache.get_tensor(outputs)

        #get input tensors
        if isinstance(inputs, list):
            in_tensors = [out_cache.get_tensor(inp) for inp in inputs]
        else:
            in_tensors = out_cache.get_tensor(inputs)

        return (in_tensors,out_tensors)


    def plot(self, save_path=None):
        print("dumping call graph...............start")
        print("===============dumpping call queue============")
        for call in self.callqueue_:
            print(call)
        print("===============dumpping tensors============")
        print(self.tensors_)
        print("dumping call graph...............end")
