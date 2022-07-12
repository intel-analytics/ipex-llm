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

from logging import warning

import pytorch_lightning.lite as lite

from bigdl.nano.pytorch.strategies import create_IPEXStrategy


class LightningLite(lite.LightningLite):
    """
    LightningLite for BigDL-Nano pytorch.

    This LightningLite extends PyTorch Lightning's LightningLite by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16: bool = False,
                 distributed_backend: str = "subprocess",
                 *args, **kwargs) -> None:
        """
        Create a LightningLite with nano acceleration.

        :param num_processes: number of processes in distributed training, defaults to 1
        :param use_ipex: whether use ipex acceleration, defaults to False
        :param enable_bf16: whether use bf16 acceleration, defaults to False
        :param distributed_backend: use which backend in distributed mode, defaults to "subprocess"
        """
        # Check keyword arguments
        if "strategy" in kwargs:
            warning(f"""strategy will be specified by bigdl-nano,
            strategy entered {kwargs['strategy']} will be ignored.""")

        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

        strategy = None
        if num_processes == 1:
            strategy = create_IPEXStrategy(enable_bf16=self.enable_bf16)
        else:
            pass

        kwargs["strategy"] = strategy
        super().__init__(*args, **kwargs)
