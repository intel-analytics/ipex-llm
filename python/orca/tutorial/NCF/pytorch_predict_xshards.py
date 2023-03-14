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
from process_xshards import get_feature_cols
from pytorch_model import NCF
from utils import *

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import XShards


# Step 1: Init Orca Context
args = parse_args("PyTorch NCF Prediction with Orca XShards", mode="predict")
init_orca(args.cluster_mode, extra_python_lib="process_xshards.py,pytorch_model.py,utils.py")


# Step 2: Load the processed data
data = XShards.load_pickle(os.path.join(args.data_dir, "test_processed_xshards"))


# Step 3: Define the model
def model_creator(config):
    model = NCF(user_num=config["user_num"],
                item_num=config["item_num"],
                factor_num=config["factor_num"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                model="NeuMF-end",
                sparse_feats_input_dims=config["sparse_feats_input_dims"],
                sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                num_dense_feats=config["num_dense_feats"])
    return model


# Step 4: Load the model
config = load_model_config(args.model_dir, "config.json")
est = Estimator.from_torch(model=model_creator,
                           config=config,
                           backend=args.backend,
                           workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, "NCF_model"))


# Step 5: Distributed inference of the loaded model
predictions = est.predict(data,
                          feature_cols=get_feature_cols(),
                          batch_size=10240)
print("Prediction results of the first 5 rows:")
print(predictions.head(n=5))


# Step 6: Save the prediction results
predictions.save_pickle(os.path.join(args.data_dir, "test_predictions_xshards"))


# Step 7: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
