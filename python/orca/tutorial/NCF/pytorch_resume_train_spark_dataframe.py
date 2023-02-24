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

from process_spark_dataframe import get_feature_cols, get_label_cols
from pytorch_model import NCF
from utils import *

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
args = parse_args("PyTorch NCF Resume Training with Spark DataFrame")
init_orca(args.cluster_mode, extra_python_lib="process_spark_dataframe.py,"
                                              "pytorch_model.py,utils.py")
spark = OrcaContext.get_spark_session()


# Step 2: Load the processed data
train_df = spark.read.parquet(os.path.join(args.data_dir, "train_processed_dataframe.parquet"))
test_df = spark.read.parquet(os.path.join(args.data_dir, "test_processed_dataframe.parquet"))


# Step 3: Define the model, optimizer and loss
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
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config["lr"])


def scheduler_creator(optimizer, config):
    return optim.lr_scheduler.StepLR(optimizer, step_size=1)

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator after loading the model
config = load_model_config(args.model_dir, "config.json")
callbacks = get_pytorch_callbacks(args)
scheduler_creator = scheduler_creator if args.lr_scheduler else None

est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=loss,
                           scheduler_creator=scheduler_creator,
                           metrics=[Accuracy(), Precision(), Recall()],
                           config=config,
                           backend=args.backend,
                           use_tqdm=True,
                           workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, "NCF_model"))

train_stats = est.fit(data=train_df,
                      epochs=2,
                      batch_size=10240,
                      feature_cols=get_feature_cols(),
                      label_cols=get_label_cols(),
                      validation_data=test_df,
                      callbacks=callbacks)
print("Train results:")
for epoch_stats in train_stats:
    for k, v in epoch_stats.items():
        print("{}: {}".format(k, v))
    print()


# Step 5: Save the trained PyTorch model
est.save(os.path.join(args.model_dir, "NCF_resume_model"))


# Step 6: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
