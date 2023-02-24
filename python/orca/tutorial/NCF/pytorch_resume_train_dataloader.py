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
import torch.utils.data as data

from pytorch_dataset import load_dataset, process_users_items, get_input_dims
from pytorch_model import NCF
from utils import *

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
args = parse_args("PyTorch NCF Resume Training with DataLoader")
init_orca(args.cluster_mode, extra_python_lib="pytorch_dataset.py,pytorch_model.py,utils.py")


# TODO: Save the processed data of the data loader as well?
# Step 2: Define train and test datasets as PyTorch DataLoader
def train_loader_func(config, batch_size):
    train_dataset, _ = load_dataset(config["data_dir"], config["dataset"], num_ng=4)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    _, test_dataset = load_dataset(config["data_dir"], config["dataset"], num_ng=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    users, items, user_num, item_num, sparse_features, dense_features, \
        total_cols = process_users_items(config["data_dir"], config["dataset"])
    sparse_feats_input_dims, num_dense_feats = get_input_dims(users, items,
                                                              sparse_features, dense_features)
    model = NCF(user_num=user_num,
                item_num=item_num,
                factor_num=config["factor_num"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                model="NeuMF-end",
                sparse_feats_input_dims=sparse_feats_input_dims,
                sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                num_dense_feats=num_dense_feats)
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

train_stats = est.fit(train_loader_func,
                      epochs=2,
                      batch_size=10240,
                      validation_data=test_loader_func,
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
