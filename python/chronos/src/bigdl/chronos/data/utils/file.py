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


def parquet2pd(path, columns=None, **kwargs):
    """
    Read a parquet file to a pandas dataframe. It supports local, s3, hdfs file systems.

    :param path: string. parquet file path
    :param columns: list. If not None, all columns will be read from the file.
    :param kwargs: Any additional kwargs are passed to the pd.read_parquet
           and pyarrow.parquet.read_table.
    """
    import pandas as pd
    df = pd.read_parquet(path, engine="pyarrow", columns=columns, **kwargs)
    return df
