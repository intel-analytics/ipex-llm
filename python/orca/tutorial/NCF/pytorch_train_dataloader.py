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

from pytorch_dataset import train_loader_func, test_loader_func, get_user_num_item_num
from pytorch_model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Define train and test datasets as PyTorch DataLoader
dataset_dir = "./ml-1m"
user_num, item_num = get_user_num_item_num(dataset_dir)


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    model = NCF(config['user_num'], config['item_num'],
                factor_num=config['factor_num'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                model=config['model'])
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config['lr'])

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
backend = "ray"  # "ray" or "spark"

est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=backend,
                           config={'user_num': user_num, 'item_num': item_num,
                                   'dataset_dir': dataset_dir,
                                   'num_ng': 4,
                                   'factor_num': 16,
                                   'num_layers': 3,
                                   'dropout': 0.5,
                                   'lr': 0.001,
                                   'model': "NeuMF-end"})
est.fit(data=train_loader_func, epochs=10, batch_size=256)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=256)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
