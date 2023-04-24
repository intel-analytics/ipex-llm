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
import warnings
from typing import List, Tuple, Callable

from bigdl.nano.utils.common import EnvContext
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import schedule_workers
from torch import nn


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
                ("preprocess", preprocess, {"cores_per_worker": 4, 'worker_num': 1}),
                ("inference", model, {"cores_per_worker": 8, 'worker_num': 1}),
            ])
        ```
        will create a pipeline which has stage "preprocess" and "inference",
        and the "preprocess" stage will call `preprocess(input)` with 4 CPU cores
        for openmp worload, while the "inference" stage will call `model(input)`
        with 8 CPU cores for openmp workload.

        :param stages: A list of configurations for each stage, each stage should consist of
            a 'name'(str), a 'function'(Callable), and a 'config'(dict).
        """
        queues = [mp.Queue() for _ in range(len(stages) + 1)]
        self.send_ = queues[0]
        self.recv_ = queues[-1]
        self.cores = schedule_workers(1)[0]
        self.ps = list(itertools.chain(
            self._launch_stage(stage, queues[i], queues[i + 1])
            for i, stage in enumerate(stages)))

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
        for idx, value in enumerate(inputs):
            self.send_.put((idx, value))
            outputs.append(None)

        for _ in range(len(outputs)):
            idx, output = self.recv_.get()
            outputs[idx] = output
        return outputs

    def _launch_stage(self, stage: Tuple, recv_: mp.Queue, send_: mp.Queue):
        name, func, config = stage

        subprocess_env = {}
        worker_num = config.get("worker_num", 1)
        invalidInputError(isinstance(worker_num, int),
                          f'worker_num config should input an int value,'
                          f' but get {type(worker_num)}')
        process_list = []

        cores_per_worker = config.get("cores_per_worker", None)
        for _ in range(worker_num):
            if isinstance(cores_per_worker, int):
                cores = self.cores[:cores_per_worker]
                self.cores = self.cores[cores_per_worker:]
                if len(cores) < cores_per_worker:
                    warnings.warn(f"stage {name} requires {cores_per_worker} cores,"
                                  f" but there are only {len(cores)} cores left")
                subprocess_env["KMP_AFFINITY"] = (
                    f"granularity=fine,proclist"
                    f"=[{','.join([str(i) for i in cores])}],explicit")
                subprocess_env["OMP_NUM_THREADS"] = str(len(cores))

            with EnvContext(env=subprocess_env):
                p = mp.Process(target=self._stage_wrapper,
                               args=(func, recv_, send_),
                               daemon=True)
                p.start()
                process_list.append(p)
        return process_list

    @staticmethod
    def _stage_wrapper(func: Callable, recv_: mp.Queue, send_: mp.Queue):
        if isinstance(func, nn.Module):
            from bigdl.nano.pytorch import InferenceOptimizer
            with InferenceOptimizer.get_context(func):
                while True:
                    idx, inputs = recv_.get()
                    outputs = func(inputs)
                    send_.put((idx, outputs))
        else:
            while True:
                idx, inputs = recv_.get()
                outputs = func(inputs)
                send_.put((idx, outputs))
