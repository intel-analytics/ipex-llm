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
import os

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy


# Step 1: Init Orca Context
sc = init_orca_context(cluster_mode="local")


# Step 2: Define train and test datasets as PyTorch DataLoader
config = dict(
    num_nodes=10,
    num_edges=50,
    dim_node_features=4,
    num_classes=5,
    factor_num=16,
    lr=0.001,
)


class MyOwnDataset(InMemoryDataset):
    def __init__(self, data):
        self.data = data
        self._num_nodes = data.num_nodes

    def __len__(self):
        return self._num_nodes

    def __getitem__(self, idx):
        return idx, self.data.x, self.data.edge_index, self.data.y[idx]


def load_dataset():
    edge_index = torch.randint(config["num_nodes"], size=(config["num_edges"], 2),
                               dtype=torch.long)
    x = torch.randn(size=(config["num_nodes"], config["dim_node_features"]),
                    dtype=torch.float)
    y = torch.randint(config["num_classes"], size=(config["num_nodes"],),
                      dtype=torch.long)
    data1 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

    train_dataset = MyOwnDataset(data1)
    test_dataset = MyOwnDataset(data1)
    return train_dataset, test_dataset


def train_loader_func(config, batch_size):
    train_dataset, _ = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    _, test_dataset = load_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(config["dim_node_features"], config["factor_num"])
        self.conv2 = GCNConv(config["factor_num"], config["num_classes"])

    def forward(self, *args):
        idx, x, edge_index = args[0], args[1][0], args[2][0]

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x[idx.tolist()]


def model_creator(config):
    model = GCN()
    model.train()
    return model


def optimizer_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=config["lr"])


# Step 4: Distributed training with Orca PyTorch Estimator
est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=torch.nn.NLLLoss(),
                           metrics=[Accuracy()],
                           config=config,
                           backend="ray",
                           use_tqdm=True,
                           workers_per_node=1)
train_stats = est.fit(train_loader_func,
                      epochs=10,
                      batch_size=config['num_nodes'],
                      validation_data=test_loader_func)
print("Train results:")
for epoch_stats in train_stats:
    for k, v in epoch_stats.items():
        print("{}: {}".format(k, v))
    print()


# Step 5: Distributed evaluation of the trained model
eval_stats = est.evaluate(test_loader_func,
                          batch_size=config['num_nodes'])
print("Evaluation results:")
for k, v in eval_stats.items():
    print("{}: {}".format(k, v))


# Step 6: Predict the graph
_, test_dataset = load_dataset()
test_loader = DataLoader(test_dataset, batch_size=config['num_nodes'],
                         shuffle=False, num_workers=0)
model = est.get_model()
model.eval()

for test_data in test_loader:
    result = model(*test_data)
    print("predictions:", result)
    print("labels:", test_data[-1])


# Step 7: Save the trained PyTorch model
est.save(os.path.join("./GNN_model"))


# Step 8: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
