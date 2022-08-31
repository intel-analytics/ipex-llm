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
from bigdl.nano.common.multiprocessing.backend import Backend
from bigdl.nano.utils.log4Error import invalidInputError
import os
from tempfile import TemporaryDirectory
from typing import Any


class HorovodBackend(Backend):

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
        import pickle
        import subprocess
        import sys

        with TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "args.pkl"), 'wb') as f:
                pickle.dump((envs,) + args, f)
            with open(os.path.join(temp_dir, "target.pkl"), 'wb') as f:
                pickle.dump(target, f)

            cwd_path = os.path.split(os.path.realpath(__file__))[0]

            p = subprocess.Popen(["horovodrun", "-np", str(nprocs), "-H", f"localhost:{nprocs}",
                                  sys.executable, f"{cwd_path}/horovod_worker.py", temp_dir])

            p.wait()

            if p.returncode != 0:
                invalidInputError(False, "horovodrun failed")

            results = []
            for i in range(nprocs):
                with open(os.path.join(temp_dir, f"history_{i}"), "rb") as f:
                    results.append(pickle.load(f))
        return results
