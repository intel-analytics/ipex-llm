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
import sys
import pickle

from pytorch_lightning.utilities.seed import reset_seed

from bigdl.nano.pytorch.plugins.ddp_subprocess import queue_dumper, queue_loader

if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = pickle.load(f)

    plugin, queue_list = args
    trainer = plugin.lightning_module.trainer
    plugin.mp_queue = queue_loader(queue_list)
    process_idx = int(os.environ["PROCESS_IDX"])

    reset_seed()
    plugin.set_world_ranks(process_idx)
    # rank_zero_only.rank = plugin.global_rank

    plugin.init_ddp_connection(plugin.global_rank, plugin.world_size)

    plugin.dist.rank = plugin.global_rank
    plugin.dist.device = plugin.root_device

    if plugin.sync_batchnorm:
        plugin.model = plugin.configure_sync_batchnorm(plugin.model)

    plugin.configure_ddp()

    plugin.model_to_device()

    plugin.barrier()
    results = trainer.run_stage()

    plugin.transfer_distrib_spawn_state_on_fit_end(results)
    if plugin.global_rank == 0:
        with open(os.path.join(temp_dir,
                               "results.pkl"), "wb") as f:
            results = queue_dumper(plugin.mp_queue)
            pickle.dump(results, f)
