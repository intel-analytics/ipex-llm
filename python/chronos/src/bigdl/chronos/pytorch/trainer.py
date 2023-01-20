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

from bigdl.nano.pytorch.trainer import Trainer as TSTrainer
from typing import Optional, List, Union, Any

class Trainer(TSTrainer):
    def __init__(self, num_processes: Optional[int] = None,
                 use_ipex: bool = False,
                 distributed_backend="subprocess",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 use_hpo=False,
                 auto_lr: Union[dict, bool] = True,
                 precision: Union[str, int] = 32,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(num_processes, use_ipex, distributed_backend,
                         cpu_for_each_process, use_hpo, False,
                         auto_lr, precision, *args, **kwargs)

