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
from typing import Sequence


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


def torch_loader_latency_calculate_helper(iterrun, baseline_time, func, model,
                                          warmup_sample, dataloader, forward_args):
    '''
    A simple helper to calculate average latency by using all the samples provided
    by a torch.utils.data.DataLoader
    '''
    from torch.utils.data import DataLoader

    check_flag = True
    if (not isinstance(dataloader, DataLoader)) and check_flag:
        print("training_data is not a torch.utils.data.DataLoader,\
              use single sample calculator instead!")
        check_flag = False
    if iterrun <= 2 and check_flag:
        print("Not enough iterations to test, use single sample calculator instead!")
        check_flag = False
    if dataloader.batch_size is not None and check_flag:
        if len(dataloader.dataset) / dataloader.batch_size <= min(iterrun, 10):
            print("Not enough batches to test, use single sample calculator instead!")
            check_flag = False
    else:
        if len(dataloader.dataset) <= min(iterrun, 10):
            print("Not enough samples to test, use single sample calculator instead!")
            check_flag = False

    if not check_flag:
        return latency_calculate_helper(iterrun, baseline_time, func, model, warmup_sample)

    # test run two times for more accurate latency
    for _ in range(2):
        func(model, warmup_sample)

    start_time = time.perf_counter()
    time_list = []
    end_flag = False
    cur_itr = 0
    while not end_flag:
        for _, input_sample in enumerate(dataloader):
            if isinstance(input_sample, Sequence):
                if len(input_sample) <= 2:
                    input_sample = input_sample[0]
                else:
                    input_sample = tuple(input_sample[:len(forward_args)])

            st = time.perf_counter()
            func(model, input_sample)
            end = time.perf_counter()
            time_list.append(end - st)
            # if three samples cost more than 4x time than baseline model, prune it
            if cur_itr == 2 and sum(time_list) > 12 * baseline_time:
                return np.mean(time_list) * 1000, False
            # at least need 10 iters and try to control calculation
            # time less than 10s
            if cur_itr + 1 >= min(iterrun, 10) and (end - start_time) > 10:
                end_flag = True
                iterrun = cur_itr + 1
                break
            # if reaching the max iteration number
            if cur_itr == iterrun - 1:
                end_flag = True
                break
            cur_itr += 1

    time_list.sort()
    # remove top and least 10% data
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000, True
