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
        if envs is not None:
            if isinstance(envs, list):
                assert nprocs == len(envs), "envs must have the same length with nprocs"
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                raise ValueError("envs must be a dict or a list of dict")

        return self.run_subprocess(target, args=args, nprocs=nprocs, envs=envs)

    def run_subprocess(self, target, args=..., nprocs=1, envs=None) -> None:
        import pickle
        import subprocess
        import sys

        temp_dir = args[0]
        with open(os.path.join(temp_dir, "train_ds_def.pkl"), 'wb') as f:
            pickle.dump(args[1], f)
        with open(os.path.join(temp_dir, "train_elem_spec.pkl"), 'wb') as f:
            pickle.dump(args[2], f)
        with open(os.path.join(temp_dir, "val_ds_def.pkl"), 'wb') as f:
            pickle.dump(args[3], f)
        with open(os.path.join(temp_dir, "val_elem_spec.pkl"), 'wb') as f:
            pickle.dump(args[4], f)
        with open(os.path.join(temp_dir, "fit_kwargs.pkl"), 'wb') as f:
            pickle.dump(args[5], f)
        with open(os.path.join(temp_dir, "target.pkl"), 'wb') as f:
            pickle.dump(target, f)

        ex_list = []
        cwd_path = os.path.split(os.path.realpath(__file__))[0]
        for i in range(nprocs):
            for key, val in os.environ.items():
                if key not in envs[i]:
                    envs[i][key] = val
            ex_list.append(subprocess.Popen([sys.executable, f"{cwd_path}/worker.py", temp_dir],
                                            env=envs[i]))
        for _, ex in enumerate(ex_list):
            ex.wait()

        results = []
        for i in range(nprocs):
            with open(os.path.join(temp_dir, f"history_{i}"), "rb") as f:
                results.append(pickle.load(f))
        return results
