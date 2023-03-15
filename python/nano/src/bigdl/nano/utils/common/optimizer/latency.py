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


import time
import numpy as np


def latency_calculate_helper(iterrun, baseline_time, func, *args):
    '''
    A simple helper to calculate average latency
    '''
    # test run two times for more accurate latency
    for i in range(2):
        func(*args)
    start_time = time.perf_counter()
    time_list = []
    for i in range(iterrun):
        st = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        time_list.append(end - st)
        # if three samples cost more than 4x time than baseline model, prune it
        if i == 2 and end - start_time > 12 * baseline_time:
            return np.mean(time_list) * 1000, False
        # at least need 10 iters and try to control calculation
        # time less than 10s
        if i + 1 >= min(iterrun, 10) and (end - start_time) > 10:
            iterrun = i + 1
            break
    time_list.sort()
    # remove top and least 10% data
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000, True
