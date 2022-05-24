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


from .space import AutoObject
from enum import Enum

from bigdl.nano.utils.log4Error import invalidInputError


class CALLTYPE(Enum):
    """Type of Function Calls."""

    LAYER_CALL = 1  # e.g. all keras layers such as Conv2D, etc.
    FUNC_CALL = 2  # such as keras.Input
    FUNC_SLICE = 3  # slices on output of a func, e.g. keras.Input(...)[:,:,]


class CallCache(object):
    """
    A data structure to cache the sequence of functional calls.

    Each autoobject contains a callcache object. The call sequences will
    be gradually passed down to the last function call.
    Internally, it containes a callqueue which collects the function calls in\
    order and a tensors table which is used to collect the resulting\
    tensors from function execution.
    """

    def __init__(self):
        """Create a Call Cache."""
        self.callqueue_ = []
        self.tensors_ = dict()
        self.skip = False

    def update_tensors(self, other):
        """Merge the tensortable from another CallCache."""
        self.tensors_.update(other.tensors_)

    def update_calls(self, other):
        """Merge the callqueues from another CallCache."""
        self.callqueue_ += other.callqueue_

    def append_call(self, caller, arguments, type):
        """Add one function call into the call queue."""
        self.callqueue_.append((type, caller, arguments))

    def add_tensor(self, node, tensor):
        """Update the resulting tensor a layer call execution."""
        self.tensors_[node] = tensor

    def get_tensor(self, n):
        """Get the resulting tensor of a layer call execution."""
        if isinstance(n, list):
            return [self.tensors_.get(one_n, None) for one_n in n]
        else:
            return self.tensors_.get(n, None)

    @property
    def calls(self):
        """Get all the call queue."""
        return self.callqueue_

    @staticmethod
    def create():
        """Create a call Cache."""
        cache = CallCache()
        return cache

    @staticmethod
    def update(arguments, current, ctype=CALLTYPE.LAYER_CALL):
        """
        Update the current autoobject's callcache from its input arguments.

        If the argument is also an autoobject, merge the callcache of
        the argument into current autoobject's callcache.

        :param arguments: input arguments of current layers
        :param current: the current autoobject
        :param ctype: the type of current call. Defaults to CALLTYPE.LAYER_CALL
        """
        def _update_cache_from_input(cache, inp):
            """Loop over all arguments to find any autoobjects\
            in input and merge down the callcache."""
            if isinstance(inp, AutoObject):
                invalidInputError(inp._callgraph is not None, "inp._callgraph cannot be none")
                input_callgraph = inp._callgraph
                # merge call graph from the input
                if not input_callgraph.skip:
                    cache.update_tensors(input_callgraph)
                    cache.update_calls(input_callgraph)
                    input_callgraph.skip = True
            elif isinstance(inp, list) or isinstance(inp, tuple):
                for item in inp:
                    _update_cache_from_input(cache, item)
            elif isinstance(inp, dict):
                for _, item in inp.items():
                    _update_cache_from_input(cache, item)
            else:
                # ignore other arguments
                pass

        cur_cache = CallCache.create()

        if ctype == CALLTYPE.LAYER_CALL or ctype == CALLTYPE.FUNC_CALL:

            _update_cache_from_input(cur_cache, arguments)
            cur_cache.append_call(current, arguments, ctype)
        elif ctype == CALLTYPE.FUNC_SLICE:
            (source, slice_args) = arguments
            _update_cache_from_input(cur_cache, source)
            cur_cache.append_call(current,
                                  arguments,
                                  CALLTYPE.FUNC_SLICE)
        else:
            invalidInputError(False, "Unexpected CallType: %s" % ctype)

        return cur_cache

    @staticmethod
    def execute(inputs, outputs, trial, backend):
        """
        Execute the function calls and construct the tensor graph.

        :param inputs: model input
        :param outputs: model outputs
        :param trial: the current trial which provides the sampled
            hyperparameters.
        """
        def _replace_autoobj(n, cache):
            if isinstance(n, AutoObject):
                new_n = cache.get_tensor(n)
            else:
                new_n = n
            return new_n

        def _process_arguments(arguments, cache):
            # TODO refactor
            if isinstance(arguments, list):
                new_arguments = [_process_arguments(
                    arg, cache) for arg in arguments]
            elif isinstance(arguments, tuple):
                lst = [_process_arguments(
                    arg, cache) for arg in arguments]
                new_arguments = tuple(lst)
            elif isinstance(arguments, dict):
                new_arguments = arguments.copy()
                for name, arg in new_arguments.items():
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
                invalidInputError(isinstance(caller, AutoObject), "caller should be AutoObject")
                instance = backend.instantiate(trial, caller)
                # the actual excution of the functional API
                out_tensor = instance(new_arguments)
            elif call_type == CALLTYPE.FUNC_SLICE:
                source, slice_args = arguments
                slice_args, slice_kwargs = slice_args
                source_tensor = out_cache.get_tensor(source)
                # the actual excution of the functional API
                out_tensor = source_tensor.__getitem__(*slice_args, **slice_kwargs)
            elif call_type == CALLTYPE.FUNC_CALL:
                # out_tensor = backend.instantiate(trial, caller)
                new_arguments = _process_arguments(
                    arguments, out_cache)
                invalidInputError(isinstance(caller, AutoObject), "caller should be AutoObject")
                # assume tensors does not exist in kwargs
                # replace only the non-kwargs with new_arguments
                # TODO revisit to validate the parent tensors in kwargs
                caller.args, caller.kwargs = new_arguments
                out_tensor = backend.instantiate(trial, caller)
            else:
                invalidInputError(False, "Unexpected CallType: %s" % type)
            out_cache.add_tensor(caller, out_tensor)
        out_tensors = out_cache.get_tensor(outputs)

        # get input tensors
        if isinstance(inputs, list):
            in_tensors = [out_cache.get_tensor(inp) for inp in inputs]
        else:
            in_tensors = out_cache.get_tensor(inputs)

        return (in_tensors, out_tensors)

    def plot(self, save_path=None):
        """Dump the call cache for debugging purpose."""
        print("dumping call cache...............start")
        print("===============dumpping call queue============")
        for call in self.callqueue_:
            print(call)
        print("===============dumpping tensors============")
        print(self.tensors_)
        print("dumping call cache...............end")
