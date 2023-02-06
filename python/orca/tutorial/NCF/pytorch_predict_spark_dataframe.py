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
import torch.nn as nn
import torch.optim as optim

from pytorch_model import NCF
from utils import *

from bigdl.orca.learn.pytorch import Estimator


# Step 1: Init Orca Context
args = parse_args("PyTorch NCF Predicting with Spark DataFrame")
init_orca(args, extra_python_lib="pytorch_model.py")
spark = OrcaContext.get_spark_session()


# Step 2: Load the processed data
df = spark.read.parquet(os.path.join(args.data_dir, "test_processed_dataframe.parquet"))


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


# Step 4: Create Orca PyTorch Estimator and load the model
config = load_model_config(args.model_dir, "config.json")
est = Estimator.from_torch(model=model_creator,
                           backend=args.backend,
                           config=config)
est.load(os.path.join(args.model_dir, "NCF_model"))


# Step 5: Distributed inference of the loaded model
predict_df = est.predict(data=df,
                         feature_cols=config["feature_cols"],
                         batch_size=10240)
print("Prediction results of the first 5 rows:")
predict_df.show(5)


# Step 6: Save the prediction results
predict_df.write.parquet(os.path.join(args.data_dir, "test_predictions_dataframe.parquet"),
                         mode="overwrite")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
