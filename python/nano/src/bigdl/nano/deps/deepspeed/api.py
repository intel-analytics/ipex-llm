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
import json
import uuid
import subprocess
from tempfile import TemporaryDirectory

import cloudpickle

from bigdl.nano.utils.common import schedule_processors

class DeepSpeedModel:
    def __init__(self, model, nproc: int = 2) -> None:
        authkey = str(uuid.uuid1())
        envs = schedule_processors(nproc)
        for i, env in enumerate(envs):
            env.update({
                "WORLD_SIZE": str(nproc),
                "LOCAL_RANK": str(i),
                "RANK": str(i),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "12345",
                "CROSS_RANK": "0",
                "CROSS_SIZE": "1",
                "LOCAL_SIZE": str(nproc),
                "AUTHKEY": authkey,
            })
            # print(env)

        with TemporaryDirectory() as temp_dir:
            # we also need to pass sys.path to subprocess
            with open(os.path.join(temp_dir, "sys_path.json"), "w") as f:
                json.dump(sys.path, f)

            from bigdl.nano.pytorch.dispatcher import _get_patch_status
            with open(os.path.join(temp_dir, "patch_status.json"), "w") as f:
                json.dump(_get_patch_status(), f)

            import torch
            torch.save(model, os.path.join(temp_dir, "model.bin"))

            import torch.multiprocessing as mp
            mp.current_process().authkey = bytes(authkey, encoding='utf-8')
            manager = mp.Manager()
            recv_queue = manager.Queue()
            send_queue = manager.Queue()
            with open(os.path.join(temp_dir, "queues.pkl"), "wb") as f:
                cloudpickle.dump((send_queue, recv_queue), f)

            current_dir = os.path.dirname(__file__)
            ps = [
                subprocess.Popen([sys.executable, os.path.join(current_dir, "worker.py"),
                                  temp_dir],
                                 env=envs[i])
                for i in range(nproc)
            ]
            _ = [recv_queue.get() for _i in range(nproc)]

        self.nproc = nproc
        self.ps = ps
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.destroyed = False

    def __call__(self, *args, **kwargs):
        for _i in range(self.nproc):
            self.send_queue.put(args)
        output = self.recv_queue.get()
        return output

    def shutdown(self):
        if not self.destroyed:
            _ = [p.kill() for p in self.ps]
            self.destroyed = True

    def __del__(self):
        self.shutdown()
