from bigdl.orca import init_orca_context, stop_orca_context

cluster_mode = "local"
if cluster_mode == "local":  # For local machine
    init_orca_context(cluster_mode="spark-submit")
elif cluster_mode == "k8s":  # For K8s cluster
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1)
elif cluster_mode == "yarn":  # For Hadoop/YARN cluster
    init_orca_context(cluster_mode="yarn", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def model_creator(config):
    model = LeNet()
    return model

def optim_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=0.001)

import torch
from torchvision import datasets, transforms

batch_size = 64
dir = '/tmp/dataset'

def train_loader_creator(config, batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    return train_loader

def test_loader_creator(config, batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False)
    return test_loader

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

est = Estimator.from_torch(log_to_driver=False, model=model_creator, optimizer=optim_creator, loss=nn.NLLLoss(), metrics=[Accuracy()], use_tqdm=True)

est.fit(data=train_loader_creator, epochs=1, batch_size=batch_size)
result = est.evaluate(data=test_loader_creator, batch_size=batch_size)
for r in result:
    print(r, ":", result[r])

est.save("mnist_model")

# stop orca context when program finishes
stop_orca_context()