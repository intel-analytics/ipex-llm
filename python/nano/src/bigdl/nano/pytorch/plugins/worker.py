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

import multiprocessing
import os
import sys

import cloudpickle

from pytorch_lightning.utilities.seed import reset_seed

from bigdl.nano.pytorch.plugins.ddp_subprocess import queue_dumper

if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = cloudpickle.load(f)

    plugin = args
    trainer = plugin.lightning_module.trainer
    process_idx = int(os.environ["PROCESS_IDX"])

    plugin.new_processes(process_idx, trainer, multiprocessing.SimpleQueue())

    if plugin.global_rank == 0:
        with open(os.path.join(temp_dir,
                               "results.pkl"), "wb") as f:
            results = queue_dumper(plugin.mp_queue)
            cloudpickle.dump(results, f)
