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
import subprocess
import sys
import copy

import logging
log = logging.getLogger(__name__)


def run_parallel(func, kwargs, n_procs):
    """
    Utility to Run a number of parallel processes.

    :param  args: the arges to be passed to subprocess
    :param num_processes: number of processes to run.
    """
    import cloudpickle
    from tempfile import TemporaryDirectory

    log.info("-" * 100)
    log.info(f"Starting {n_procs} parallel processes")
    log.info("-" * 100)

    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "search_kwargs.pkl"), 'wb') as f:
            cloudpickle.dump(kwargs, f)
        with open(os.path.join(temp_dir, "search_func.pkl"), 'wb') as f:
            cloudpickle.dump(func, f)

        processes = _run_subprocess(temp_dir, n_procs)

        for _, process in enumerate(processes):
            process.wait()


def _run_subprocess(tmpdir, num_processes):
    from bigdl.nano.utils.common import schedule_processors

    envs = schedule_processors(num_processes)

    processes = []
    cwd_path = os.path.split(os.path.realpath(__file__))[0]
    for i in range(num_processes):
        processes.append(subprocess.Popen([sys.executable, f"{cwd_path}/parallel_worker.py",
                                           tmpdir], env=envs[i]))

    return processes
