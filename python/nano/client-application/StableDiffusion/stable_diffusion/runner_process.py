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

import multiprocessing as mp
from multiprocessing.pool import Pool

from stable_diffusion.runner import StableDiffusionRunner

cpu_binding = False
pool = None
var_dict = None

class RunnerProcess(mp.Process):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.runner = StableDiffusionRunner.initialize() # while calling this everything in this line must be serializable
    # can not override run func here otherwise it would loop infinitely between __init__() and run()

class RunnerPool(Pool):
    def __init__(self) -> None:
        super().__init__(1)
    
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return RunnerProcess(*args, **kwargs)
    
def test_func(x):
    print("Running test_func")
    p = mp.current_process()
    y = x * x if p.runner is None else x
    print(y)


if __name__ == '__main__':    
    # test functionality
    
    with RunnerPool() as pool:
        for i in range(3):
            print(f"Applying {i} to pool")
            pool.apply(test_func, (i,))
