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
import ray
from bigdl.nano.utils.log4Error import invalidInputError


class RayBackend(Backend):

    def setup(self) -> None:
        ray.init()

    def shutdown(self) -> None:
        ray.shutdown()

    def run(self, target, args=..., nprocs=1, envs=None):
        if envs is not None:
            if isinstance(envs, list):
                invalidInputError(nprocs == len(envs),
                                  "envs must have the same length with nprocs")
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                invalidInputError(False, "envs must be a dict or a list of dict")

        results = []
        for i in range(nprocs):
            runtime_env = {
                "env_vars": envs[i]
            }
            results.append(ray.remote(target).options(runtime_env=runtime_env).remote(*args))
        return ray.get(results)
