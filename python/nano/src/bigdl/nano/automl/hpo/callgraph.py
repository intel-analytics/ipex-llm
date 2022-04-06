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
        import networkx as nx
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
    def plot(g, save_path="callgraph_plot.png"):
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
    """A data structure to cache the sequence of functional calls.
    """
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
    def update(arguments, current, ctype=CALLTYPE.LAYER_CALL):
        def update_cache_from_input(cache, inp):
            '''loop over all arguments to find any autoobjects
            in input and merge down the callcache'''
            if isinstance(inp, AutoObject):
                assert(inp._callgraph is not None)
                input_callgraph = inp._callgraph
                # merge call graph from the input
                if not input_callgraph.skip:
                    cache.update_tensors(input_callgraph)
                    cache.update_calls(input_callgraph)
                    input_callgraph.skip = True
            elif isinstance(inp, list) or isinstance(inp, tuple):
                for item in inp:
                    update_cache_from_input(cache,item)
            elif isinstance(inp, dict):
                for _, item in inp.items():
                    update_cache_from_input(cache,item)
            else:
                # ignore other arguments
                pass

        cur_cache = CallCache.create()

        if ctype == CALLTYPE.LAYER_CALL or ctype == CALLTYPE.FUNC_CALL:

            update_cache_from_input(cur_cache,arguments)
            cur_cache.append_call(current, arguments, ctype)
        elif ctype == CALLTYPE.FUNC_SLICE:
            (source, slice_args) = arguments
            update_cache_from_input(cur_cache, source)
            cur_cache.append_call(current,
                                  arguments,
                                  CALLTYPE.FUNC_SLICE)
        else:
            raise ValueError("Unexpected CallType: %s" % ctype)

        return cur_cache


    @staticmethod
    def execute(inputs, outputs, trial):
        def _replace_autoobj(n, cache):
            if isinstance(n, AutoObject):
                new_n = cache.get_tensor(n)
            else:
                new_n = n
            return new_n
        def _process_arguments(arguments, cache):
            # TODO refactor
            if isinstance(arguments, list) :
                new_arguments = [_process_arguments(
                    arg, cache) for arg in arguments]
            elif isinstance(arguments, tuple):
                lst = [_process_arguments(
                    arg, cache) for arg in arguments]
                new_arguments=tuple(lst)
            elif isinstance(arguments, dict):
                new_arguments = arguments.copy()
                for name,arg in new_arguments.items():
                    new_arg = _process_arguments(
                        arg, cache)
                    new_arguments[name] = new_arg
            else:
                new_arguments = _replace_autoobj(arguments, cache)
            return new_arguments

        out_cache = outputs._callgraph
        # get output tensors
        for call_type, caller, arguments in out_cache.calls:
            if call_type == CALLTYPE.LAYER_CALL:
                new_arguments = _process_arguments(
                    arguments, out_cache)
                # layer is an auto object
                assert(isinstance(caller, AutoObject))
                instance = OptunaBackend.instantiate(trial, caller)
                # the actual excution of the functional API
                out_tensor = instance(new_arguments)
            elif call_type == CALLTYPE.FUNC_SLICE:
                source, slice_args = arguments
                slice_args, slice_kwargs =slice_args
                source_tensor = out_cache.get_tensor(source)
                # the actual excution of the functional API
                out_tensor = source_tensor.__getitem__(*slice_args, **slice_kwargs)
            elif call_type == CALLTYPE.FUNC_CALL:
                # out_tensor = OptunaBackend.instantiate(trial, caller)
                new_arguments = _process_arguments(
                    arguments, out_cache)
                assert(isinstance(caller, AutoObject))
                # assume tensors does not exist in kwargs
                # replace only the non-kwargs with new_arguments
                # TODO revisit to validate the parent tensors in kwargs
                caller.args, caller.kwargs = new_arguments
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
        print("dumping call cache...............start")
        print("===============dumpping call queue============")
        for call in self.callqueue_:
            print(call)
        print("===============dumpping tensors============")
        print(self.tensors_)
        print("dumping call cache...............end")
