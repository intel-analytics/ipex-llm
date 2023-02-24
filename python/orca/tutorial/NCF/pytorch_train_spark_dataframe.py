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

from process_spark_dataframe import prepare_data
from pytorch_model import NCF
from utils import *

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
args = parse_args("PyTorch NCF Training with Spark DataFrame")
init_orca(args.cluster_mode, extra_python_lib="process_spark_dataframe.py,"
                                              "pytorch_model.py,utils.py")


# Step 2: Read and process data using Spark DataFrame
train_df, test_df, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data(args.data_dir, args.dataset, neg_scale=4)


# Step 3: Define the model, optimizer and loss
config = dict(
    user_num=user_num,
    item_num=item_num,
    factor_num=16,
    num_layers=3,
    dropout=0.5,
    lr=0.01,
    sparse_feats_input_dims=sparse_feats_input_dims,
    sparse_feats_embed_dims=8,
    num_dense_feats=num_dense_feats
)


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


# Step 4: Distributed training with Orca PyTorch Estimator
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
train_stats = est.fit(train_df,
                      epochs=2,
                      batch_size=10240,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      validation_data=test_df,
                      callbacks=callbacks)
print("Train results:")
for epoch_stats in train_stats:
    for k, v in epoch_stats.items():
        print("{}: {}".format(k, v))
    print()


# Step 5: Distributed evaluation of the trained model
eval_stats = est.evaluate(test_df,
                          feature_cols=feature_cols,
                          label_cols=label_cols,
                          batch_size=10240)
print("Evaluation results:")
for k, v in eval_stats.items():
    print("{}: {}".format(k, v))


# Step 6: Save the trained PyTorch model and processed data for resuming training or prediction
est.save(os.path.join(args.model_dir, "NCF_model"))
save_model_config(config, args.model_dir, "config.json")
train_df.write.parquet(os.path.join(args.data_dir,
                                    "train_processed_dataframe.parquet"), mode="overwrite")
test_df.write.parquet(os.path.join(args.data_dir,
                                   "test_processed_dataframe.parquet"), mode="overwrite")


# Step 7: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
