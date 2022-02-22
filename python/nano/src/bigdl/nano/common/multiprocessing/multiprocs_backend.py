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

import copy
import os
from itertools import product
import multiprocessing
from multiprocessing import Process, Pool

from bigdl.nano.common.multiprocessing.backend import Backend


class MultiprocessingBackend(Backend):
    def __init__(self):
        self._pool = None

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def run(self, target, args=..., nprocs=1, envs=None) -> None:
        self.run_pool(target, args=args, nprocs=nprocs, envs=envs)
        # self.run_process(target, args=args, nprocs=nprocs, envs=envs)

    def run_process(self, target, args=..., nprocs=1, envs=None) -> None:
        if envs is not None:
            if isinstance(envs, list):
                assert nprocs == len(envs), "envs must have the same length with nprocs"
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                raise ValueError("envs must be a dict or a list of dict")

        proc_list = []
        for i in range(nprocs):
            env_back, env_del_list = dict(), list()
            for key, value in envs[i].items():
                if key in os.environ:
                    env_back[key] = os.environ[key]
                else:
                    env_del_list.append(key)
                os.environ[key] = value

            p = Process(target=target, args=args)
            p.start()
            proc_list.append(p)

        for p in proc_list:
            p.join()
            print(f"process {i} exitcode: {p.exitcode}")
            for key, value in env_back.items():
                os.environ[key] = value
            for _, key in enumerate(env_del_list):
                del os.environ[key]

    def run_pool(self, target, args=..., nprocs=1, envs=None) -> None:
        self._pool = Pool(processes=nprocs)

        if envs is not None:
            if isinstance(envs, list):
                assert nprocs == len(envs), "envs must have the same length with nprocs"
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                raise ValueError("envs must be a dict or a list of dict")

        result = []
        for i in range(nprocs):
            env_back, env_del_list = dict(), list()
            for key, value in envs[i].items():
                if key in os.environ:
                    env_back[key] = os.environ[key]
                else:
                    env_del_list.append(key)
                os.environ[key] = value

            res = self._pool.apply_async(func=target, args=list(args)+[envs[i]])
            result.append(res)

            for key, value in env_back.items():
                os.environ[key] = value
            for _, key in enumerate(env_del_list):
                del os.environ[key]

        self._pool.close()
        self._pool.join()
        for res in result:
            print(res.get())
