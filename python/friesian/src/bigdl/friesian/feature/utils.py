#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.common.utils import callZooFunc


def compute(df):
    return callZooFunc("float", "compute", df)


def log_with_clip(df, columns, clip=True):
    return callZooFunc("float", "log", df, columns, clip)


def generate_string_idx(df, columns, freq_limit):
    return callZooFunc("float", "generateStringIdx", df, columns, freq_limit)


def fill_na(df, fill_val, columns):
    return callZooFunc("float", "fillNa", df, fill_val, columns)


def fill_na_int(df, fill_val, columns):
    return callZooFunc("float", "fillNaInt", df, fill_val, columns)


def clip_min(df, columns, min):
    return callZooFunc("float", "clipMin", df, columns, min)
