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

import pandas as pd
import os

if __name__ == "__main__":
    data_dir = "path/to/ml-1m"

    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::', header=None,
                          names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin-1')
    users = pd.read_csv(os.path.join(data_dir, 'users.dat'), sep='::', header=None,
                        names=["user_id", "gender", "age", "occupation", "zip_code"],
                        encoding='latin-1')
    total = ratings.join(users.set_index("user_id"), on="user_id")
    total.to_parquet(os.path.join(data_dir, "total.parquet"))
