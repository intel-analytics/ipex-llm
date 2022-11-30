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

# Step 0: Import necessary libraries
import math

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator

from tf_model import ncf_model

# Step 1: Init Orca Context
init_orca_context(memory='4g')
spark = OrcaContext.get_spark_session()

# Step 2: Load the model and data
est = Estimator.from_keras()
est.load('NCF_model')
data = spark.read.parquet('test_dataframe')

# Step 3: Define the input feature columns
sparse_features = ['zipcode', 'gender', 'category', 'occupation']
dense_features = ['age']
feature_cols = ['user', 'item'] + sparse_features + dense_features

# Step 4: Predict the result
res = est.predict(
    data,
    batch_size=256,
    steps=math.ceil(data.count() / 256),
    feature_cols=feature_cols
)

# Step 5: Save the prediction result
res.write.parquet('predict_result', mode='overwrite')

# Step 6: Stop Orca Context when program finishes
stop_orca_context()
