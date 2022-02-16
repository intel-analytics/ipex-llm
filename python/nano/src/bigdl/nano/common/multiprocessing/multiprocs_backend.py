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
from itertools import product
from multiprocessing import Pool

from bigdl.nano.common.multiprocessing.backend import Backend


class MultiprocessingBackend(Backend):

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

        for key, value in envs[0].items():
            os.environ[key] = value

        args = [args] * nprocs
        with Pool(nprocs) as p:
            results = p.starmap(target, args)

        return results
