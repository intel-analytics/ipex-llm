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


import os
from multiprocessing import Process
import multiprocessing as mp

from bigdl.nano.utils.common import invalidInputError


class _append_return_to_pipe(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, queue, *args, **kwargs):
        res = self.func(*args, **kwargs)
        queue.put(res)


class spawn_new_process(object):
    def __init__(self, func):
        '''
        This class is to decorate on another function where you want to run in a new process
        with brand new environment variables (e.g. OMP/KMP, LD_PRELOAD, ...)
        an example to use this is
            ```python
            def throughput_helper(model, x):
                st = time.time()
                for _ in range(100):
                    model(x)
                return time.time() - st

            # make this wrapper
            # note: please name the new function a new func name.
            new_throughput_helper = spawn_new_process(throughput_helper)

            # this will run in current process
            duration = new_throughput_helper(model, x)

            # this will run in a new process with new env var effective
            duration = new_throughput_helper(model, x, env_var={"OMP_NUM_THREADS": "1",
                                                                "LD_PRELOAD": ...})
            ```
        '''
        self.func = func

    def __call__(self, *args, **kwargs):
        if "env_var" in kwargs:
            # check env_var should be a dict
            invalidInputError(isinstance(kwargs['env_var'], dict),
                              "env_var should be a dict")

            # prepare
            # 1. save the original env vars
            # 2. set new ones
            # 3. change the backend of multiprocessing from fork to spawn
            # 4. delete "env_var" from kwargs
            old_env_var = {}
            for key, value in kwargs['env_var'].items():
                old_env_var[key] = os.environ.get(key, "")
                os.environ[key] = value
            try:
                mp.set_start_method('spawn')
            except Exception:
                pass
            del kwargs["env_var"]

            # new process
            # block until return
            q = mp.Queue()
            new_func = _append_return_to_pipe(self.func)
            p = Process(target=new_func, args=(q,) + args, kwargs=kwargs)
            p.start()
            return_val = q.get()
            p.join()

            # recover
            for key, value in old_env_var.items():
                os.environ[key] = value

            return return_val
        else:
            return self.func(*args, **kwargs)
