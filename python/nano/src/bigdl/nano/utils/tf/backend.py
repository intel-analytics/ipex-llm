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
from tempfile import TemporaryDirectory
from typing import Any

from bigdl.nano.utils.common import Backend
from bigdl.nano.utils.common import invalidInputError


class MultiprocessingBackend(Backend):

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def run(self, target, args=..., nprocs=1, envs=None) -> Any:
        if envs is not None:
            if isinstance(envs, list):
                invalidInputError(nprocs == len(envs),
                                  "envs must have the same length with nprocs")
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                invalidInputError(False, "envs must be a dict or a list of dict")

        return self.run_subprocess(target, args=args, nprocs=nprocs, envs=envs)

    def run_subprocess(self, target, args=..., nprocs=1, envs=None) -> Any:
        import cloudpickle
        import subprocess
        import sys

        with TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "args.pkl"), 'wb') as f:
                cloudpickle.dump(args, f)
            with open(os.path.join(temp_dir, "target.pkl"), 'wb') as f:
                cloudpickle.dump(target, f)

            ex_list = []
            cwd_path = os.path.dirname(__file__)
            for i in range(nprocs):
                for key, val in os.environ.items():
                    if key not in envs[i]:
                        envs[i][key] = val
                if 'OMP_NUM_THREADS' in envs[i]:
                    set_op_thread = envs[i]['OMP_NUM_THREADS']
                else:
                    # 0 means the system picks an appropriate number
                    set_op_thread = '0'
                ex_list.append(subprocess.Popen([sys.executable, f"{cwd_path}/subprocess_worker.py",
                                                 temp_dir, set_op_thread], env=envs[i]))
            for _, ex in enumerate(ex_list):
                ex.wait()

            results = []
            for i in range(nprocs):
                with open(os.path.join(temp_dir, f"history_{i}"), "rb") as f:
                    results.append(cloudpickle.load(f))
        return results
