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
import shutil
import tempfile
from bigdl.chronos.data.utils.file import parquet2pd
import pandas as pd
import numpy as np


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


class TestFile:

    def test_parquet2pd_local(self):
        temp = tempfile.mkdtemp()
        try:
            path = os.path.join(temp, "test.parquet")
            df = get_ts_df()
            df.to_parquet(path)
            df_from_parquet = parquet2pd(path)
            pd.testing.assert_frame_equal(df, df_from_parquet)
        finally:
            shutil.rmtree(temp)
