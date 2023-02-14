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

import itertools
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple, Dict, Callable

from torch import nn


class Pipeline():
    def __init__(self, stages: List[Tuple]):
        conns = [reversed(mp.Pipe(duplex=False)) for _i in range(len(stages) + 1)]
        conns = list(itertools.chain(*conns))
        self.send_ = conns[0]
        self.recv_ = conns[-1]
        self.ps = [
            self._run_stage(stage, conns[i * 2 + 1], conns[i * 2 + 2])
            for i, stage in enumerate(stages)
        ]

    def run(self, inputs):
        for i in inputs:
            self.send_.send(i)

        output_list = []
        for i in inputs:
            output = self.recv_.recv()
            output_list.append(output)

        return output_list

    def _run_stage(self, stage: Tuple, recv_: Connection, send_: Connection):
        name, func, config = stage
        self._validate_name(name)
        self._validate_config(config)

        p = mp.Process(target=self._stage_wrapper,
                       args=(func, recv_, send_),
                       daemon=True)
        p.start()
        return p

    @staticmethod
    def _stage_wrapper(func: Callable, recv_: Connection, send_: Connection):
        if isinstance(func, nn.Module):
            from bigdl.nano.pytorch import InferenceOptimizer
            with InferenceOptimizer.get_context(func):
                while True:
                    inputs = recv_.recv()
                    outputs = func(inputs)
                    send_.send(outputs)
        else:
            while True:
                inputs = recv_.recv()
                outputs = func(inputs)
                send_.send(outputs)

    @staticmethod
    def _validate_name(name: str):
        pass

    @staticmethod
    def _validate_config(config: Dict):
        pass

    @staticmethod
    def _apply_config(config: Dict):
        pass
