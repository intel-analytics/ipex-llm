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
from bigdl.orca.data import XShards
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import init_orca_context, stop_orca_context

from process_xshards import get_feature_cols


# Step 1: Init Orca Context
init_orca_context(cluster_mode="local")


# Step 2: Load the model and data
est = Estimator.from_keras()
est.load("NCF_model")
data = XShards.load_pickle("./train_processed_xshards")
feature_cols = get_feature_cols()


# Step 3: Distributed inference of the loaded model
predictions = est.predict(data, batch_size=10240, feature_cols=feature_cols)


# Step 4: Save the prediction results
predictions.save_pickle("test_predictions_xshards")


# Step 5: Stop Orca Context when program finishes
stop_orca_context()
