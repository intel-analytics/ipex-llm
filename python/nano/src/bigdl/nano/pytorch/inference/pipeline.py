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
import warnings
import itertools
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple, Dict, Callable

from torch import nn

from bigdl.nano.utils.common import schedule_workers
from bigdl.nano.utils.common import EnvContext


class Pipeline():
    """
    A two-stage pipeline, which will run all stages in parallel.

    A simple usage:
    ```
        pipeline = Pipeline([
            ("preprocess", preprocess, {}),
            ("inference", model, {}),
        ])
        outputs = pipeline.run(inputs)
    ```
    """
    def __init__(self, stages: List[Tuple]):
        """
        Create a pipeline with given stages.

        For example,
        ```
            pipeline = Pipeline([
                ("preprocess", preprocess, {"core_num": 4}),
                ("inference", model, {"core_num": 8}),
            ])
        ```
        will create a pipeline which has stage "preprocess" and "inference",
        and the "preprocess" stage will call `preprocess(input)` with 4 CPU cores,
        while the "inference" stage will call `model(input)` with 8 CPU cores.

        :param stages: A list of configurations for each stage, each stage should consist of
            a 'name'(str), a 'function'(Callable), and a 'config'(dict).
        """
        conns = [reversed(mp.Pipe(duplex=False)) for _i in range(len(stages) + 1)]
        conns = list(itertools.chain(*conns))
        self.send_ = conns[0]
        self.recv_ = conns[-1]
        self.cores = schedule_workers(1)[0]
        self.ps = [
            self._launch_stage(stage, conns[i * 2 + 1], conns[i * 2 + 2])
            for i, stage in enumerate(stages)
        ]

    def run(self, inputs):
        """
        Conduct inference on the inputs in a pipeline.

        For example,
        ```
            outputs = pipeline.run(inputs)
        ```
        is equivalent to
        ```
            outputs = [model(preprocess(i)) for i in inputs]
        ```
        except that the pipeline will run all stages in parallel
        and handle inference automatically.

        :param inputs: A list of batches.
        :return: A list of the result of inference.
        """
        outputs = []
        for i in inputs:
            self.send_.send(i)
            outputs.append(None)

        outputs = [self.recv_.recv() for _ in outputs]
        return outputs

    def _launch_stage(self, stage: Tuple, recv_: Connection, send_: Connection):
        name, func, config = stage

        subprocess_env = {}
        if isinstance(config.get("core_num", None), int):
            core_num = config["core_num"]
            cores = self.cores[:core_num]
            self.cores = self.cores[core_num:]
            if len(cores) < core_num:
                warnings.warn(f"stage {name} requires {core_num} cores,"
                              f" but there are only {len(cores)} cores left")
            subprocess_env["KMP_AFFINITY"] = (f"granularity=fine,proclist"
                                              f"=[{','.join([str(i) for i in cores])}],explicit")
            subprocess_env["OMP_NUM_THREADS"] = str(len(cores))

        with EnvContext(env=subprocess_env):
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
