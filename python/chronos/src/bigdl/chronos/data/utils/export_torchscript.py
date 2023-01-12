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

import torch
import torch.nn as nn
from typing import List
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from .utils import _to_list


class ExportJIT(nn.Module):
    def __init__(self, lookback: int, id_index: int,
                 target_feature_index: List[int],
                 operation: str) -> None:
        super().__init__()
        self.lookback = lookback
        self.target_feature_index = target_feature_index
        self.id_index = id_index
        self.operation = operation

    def _shift(self, data, i: int):
        res = torch.empty_like(data, dtype=torch.float32)
        if i < 0:
            res[:] = torch.roll(data, i, 0)
            res[i:] = torch.nan
        elif i == 0:
            res[:] = data
        return res

    def _groupby(self, data, colunm):
        # group the data using unique values in column
        # This code is adapted from https://github.com/pytorch/pytorch/issues/20613

        sorted_values, order = colunm.sort(0)
        delta = colunm[1:] - colunm[:-1]
        non_zero = delta.nonzero()

        if len(non_zero) == 0:
            return [data]

        cutpoints = non_zero[0]
        cutpoints: List[int] = cutpoints.add(1).tolist()
        res: List[torch.Tensor] = []
        for start, end in zip([0] + cutpoints, cutpoints + [len(colunm)]):
            index, _ = order[start:end].sort(0)
            res.append(data[index])

        return res

    def _roll_tensor(self, data, lookback: int, target_feature_index: List[int]):
        data = data[:, target_feature_index]
        data = torch.unsqueeze(data, 1)
        roll_data = torch.cat([self._shift(data, i) for i in range(0, -lookback, -1)],
                              dim=1)
        window_idx = torch.arange(lookback)
        if data.size()[0] >= lookback:
            roll_data = roll_data[:data.size()[0]-lookback+1, window_idx, :]
        else:
            roll_data = roll_data[:0, window_idx, :]
        return roll_data

    def roll(self, data, lookback: int, id_index: int, target_feature_index: List[int]):
        id_col = data[:, id_index]
        res: List[torch.Tensor] = self._groupby(data, id_col)
        roll_result: List[torch.Tensor] = []
        for group in res:
            roll_result.append(self._roll_tensor(group, lookback, target_feature_index))

        return torch.cat(roll_result, dim=0)

    def scale(self, data):
        return data

    def unscale(self, data):
        return data

    def export_preprocessing(self, data):
        data[:, self.target_feature_index] = self.scale(data[:, self.target_feature_index])
        data_roll = self.roll(data, self.lookback, self.id_index, self.target_feature_index)
        return data_roll

    def export_postprocessing(self, data):
        return self.unscale(data)

    def forward(self, data):
        if self.operation == "preprocessing":
            return self.export_preprocessing(data)
        elif self.operation == "postprocessing":
            return self.export_postprocessing(data)
        else:
            # never reached here
            return data


class ExportWithStandardScaler(ExportJIT):
    def __init__(self, scaler: StandardScaler, lookback: int,
                 id_index: int, target_feature_index: List[int],
                 scaler_index: List[int], operation: str) -> None:
        super().__init__(lookback, id_index, target_feature_index, operation)
        self.scale_ = torch.from_numpy(scaler.scale_).type(torch.float64)
        self.mean_ = torch.from_numpy(scaler.mean_).type(torch.float64)
        self.with_mean: bool = bool(scaler.with_mean)
        self.with_std: bool = bool(scaler.with_std)
        self.lookback: int = lookback
        self.id_index: int = id_index
        self.target_feature_index = target_feature_index
        self.scaler_index = scaler_index

    def scale(self, data):
        data_scale = torch.zeros(data.size(), dtype=torch.float64)
        for i in range(data.size()[1]):
            value_mean = self.mean_[i] if self.with_mean \
                else torch.zeros(self.mean_.size(), dtype=torch.float64)
            value_scale = self.scale_[i] if self.with_std \
                else torch.ones(self.scale_.size(), dtype=torch.float64)
            data_scale[:, i] = (data[:, i] - value_mean) / value_scale
        return data_scale

    def unscale(self, data):
        data_unscale = torch.zeros(data.size(), dtype=torch.float64)
        for i in self.scaler_index:
            value_mean = self.mean_[i] if self.with_mean \
                else torch.zeros(self.mean_.size(), dtype=torch.float64)
            value_scale = self.scale_[i] if self.with_std \
                else torch.ones(self.scale_.size(), dtype=torch.float64)
            data_unscale[:, :, i] = data[:, :, i] * value_scale + value_mean
        return data_unscale


class ExporWithMaxAbsScaler(ExportJIT):
    def __init__(self, scaler: MaxAbsScaler, lookback: int,
                 id_index: int, target_feature_index: List[int],
                 scaler_index: List[int], operation: str) -> None:
        super().__init__(lookback, id_index, target_feature_index, operation)
        self.scale_ = torch.from_numpy(scaler.scale_).type(torch.float64)
        self.lookback: int = lookback
        self.id_index = id_index
        self.target_feature_index = target_feature_index
        self.scaler_index = scaler_index

    def scale(self, data):
        data_scale = torch.zeros(data.size(), dtype=torch.float64)
        for i in range(data.size()[1]):
            value_max_abs = self.scale_[i]
            data_scale[:, i] = data[:, i] / value_max_abs
        return data_scale

    def unscale(self, data):
        data_unscale = torch.zeros(data.size(), dtype=torch.float64)
        for i in self.scaler_index:
            value_max_abs = self.scale_[i]
            data_unscale[:, :, i] = data[:, :, i] * value_max_abs
        return data_unscale


class ExportWithMinMaxScaler(ExportJIT):
    def __init__(self, scaler: MinMaxScaler, lookback: int,
                 id_index: int, target_feature_index: List[int],
                 scaler_index: List[int], operation: str) -> None:
        super().__init__(lookback, id_index, target_feature_index, operation)
        self.scale_ = torch.from_numpy(scaler.scale_).type(torch.float64)
        self.min_ = torch.from_numpy(scaler.min_).type(torch.float64)
        self.lookback: int = lookback
        self.id_index: int = id_index
        self.target_feature_index = target_feature_index
        self.scaler_index = scaler_index

    def scale(self, data):
        data_scale = torch.zeros(data.size(), dtype=torch.float64)
        for i in range(data.size()[1]):
            value_min = self.min_[i]
            value_scale = self.scale_[i]
            data_scale[:, i] = data[:, i] * value_scale + value_min
        return data_scale

    def unscale(self, data):
        data_unscale = torch.zeros(data.size(), dtype=torch.float64)
        for i in self.scaler_index:
            value_min = self.min_[i]
            value_scale = self.scale_[i]
            data_unscale[:, :, i] = (data[:, :, i] - value_min) / value_scale
        return data_unscale


class ExportWithRobustScaler(ExportJIT):
    def __init__(self, scaler: RobustScaler, lookback: int,
                 id_index: int, target_feature_index: List[int],
                 scaler_index: List[int], operation: str) -> None:
        super().__init__(lookback, id_index, target_feature_index, operation)
        self.scale_ = torch.from_numpy(scaler.scale_).type(torch.float64)
        self.center_ = torch.from_numpy(scaler.center_).type(torch.float64)
        self.with_centering: bool = bool(scaler.with_centering)
        self.with_scaling: bool = bool(scaler.with_scaling)
        self.lookback: int = lookback
        self.id_index: int = id_index
        self.target_feature_index = target_feature_index
        self.scaler_index = scaler_index

    def scale(self, data):
        data_scale = torch.zeros(data.size(), dtype=torch.float64)
        for i in range(data.size()[1]):
            value_center = self.center_[i] if self.with_centering \
                else torch.zeros(self.center_.size(), dtype=torch.float64)
            value_scale = self.scale_[i] if self.with_scaling \
                else torch.ones(self.scale_.size(), dtype=torch.float64)
            data_scale[:, i] = (data[:, i] - value_center) / value_scale
        return data_scale

    def unscale(self, data):
        data_unscale = torch.zeros(data.size(), dtype=torch.float64)
        for i in self.scaler_index:
            value_center = self.center_[i] if self.with_centering \
                else torch.zeros(self.center_.size(), dtype=torch.float64)
            value_scale = self.scale_[i] if self.with_scaling \
                else torch.ones(self.scale_.size(), dtype=torch.float64)
            data_unscale[:, :, i] = data[:, :, i] * value_scale + value_center
        return data_unscale


SCALE_JIT_HELPER_MAP = {StandardScaler: ExportWithStandardScaler,
                        MaxAbsScaler: ExporWithMaxAbsScaler,
                        MinMaxScaler: ExportWithMinMaxScaler,
                        RobustScaler: ExportWithRobustScaler}


def export_processing_to_jit(scaler, lookback, id_index, target_feature_index,
                             scaler_index, operation):
    export_class = SCALE_JIT_HELPER_MAP[type(scaler)]
    return torch.jit.script(export_class(scaler, lookback,
                                         id_index, target_feature_index,
                                         scaler_index, operation))


def get_index(df, id_col, target_col, feature_col):
    id_index = df.columns.tolist().index(id_col)
    target_col = _to_list(target_col, "target_col", deploy_mode=True)
    feature_col = _to_list(feature_col, "feature_col", deploy_mode=True)

    # index of target col and feature col, will be used in scale and roll
    target_feature_index = [df.columns.tolist().index(i) for i in target_col + feature_col]
    return id_index, target_feature_index


def get_processing_module_instance(scaler, lookback, id_index, target_feature_index,
                                   scaler_index, operation):
    export_class = SCALE_JIT_HELPER_MAP[type(scaler)]
    return export_class(scaler, lookback,
                        id_index, target_feature_index,
                        scaler_index, operation)
