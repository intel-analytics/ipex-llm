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

import math


def recalculate_n_sampling(search_space, n_sampling):
    """
    Only process n_sampling.

    :param n_sampling: Number of trials to evaluate in total.
    """
    search_count = [len(v['grid_search']) for _, v in search_space.items() if isinstance(v, dict)]
    assist_num = 1
    if search_count:
        for val in search_count:
            assist_num *= val

    n_sampling /= assist_num
    # TODO Number of threads specified by the user corresponds to n_sampling and give warning.
    return math.ceil(n_sampling)


def check_quantize_available(model):
    from bigdl.chronos.model.Seq2Seq_pytorch import LSTMSeq2Seq
    from bigdl.nano.utils.log4Error import invalidInputError
    if isinstance(model, LSTMSeq2Seq):
        invalidInputError(False, f"This {model.__class__} has not supported quantization.")


def preprocess_quantize_data(self, data):
    import torch
    import types
    from torch.utils.data import DataLoader, TensorDataset
    from bigdl.chronos.data import TSDataset
    if isinstance(data, tuple):
        data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                        torch.from_numpy(data[1])))
    if isinstance(data, types.FunctionType):
        data = data(self._best_config)
    if isinstance(data, TSDataset):
        data_x, data_y = self._tsdataset_to_numpy(data, is_predict=False)
        data = DataLoader(TensorDataset(torch.from_numpy(data_x),
                                        torch.from_numpy(data_y)))
    return data
