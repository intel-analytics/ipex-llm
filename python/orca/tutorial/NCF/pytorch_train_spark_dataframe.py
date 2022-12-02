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

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Read and process data using Orca XShards
dataset_dir = "./ml-1m"
train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data("ml-1m", neg_scale=4)


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    model = NCF(user_num=config['user_num'],
                item_num=config['item_num'],
                factor_num=config['factor_num'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                model=config['model'],
                sparse_feats_input_dims=config['sparse_feats_input_dims'],
                sparse_feats_embed_dims=config['sparse_feats_embed_dims'],
                num_dense_feats=config['num_dense_feats'])
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config['lr'])

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
backend = "spark"  # "ray" or "spark"

est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=backend,
                           config={'dataset_dir': dataset_dir,
                                   'user_num': user_num,
                                   'item_num': item_num,
                                   'factor_num': 16,
                                   'num_layers': 3,
                                   'dropout': 0.5,
                                   'lr': 0.001,
                                   'model': "NeuMF-end",
                                   'sparse_feats_input_dims': sparse_feats_input_dims,
                                   'sparse_feats_embed_dims': 8,
                                   'num_dense_feats': num_dense_feats})
est.fit(data=train_data, epochs=10,
        feature_cols=feature_cols,
        label_cols=label_cols,
        batch_size=256)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_data,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      batch_size=256)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
