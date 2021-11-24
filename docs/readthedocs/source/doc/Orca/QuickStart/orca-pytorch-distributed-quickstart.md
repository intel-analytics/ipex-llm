# Use `torch.distributed` in Orca

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/pytorch_distributed_lenet_mnist.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/pytorch_distributed_lenet_mnist.ipynb)

---

**In this guide we will describe how to scale out _PyTorch_ programs using the `torch.distributed` package in Orca.**

### **Step 0: Prepare Environment**

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is needed to prepare the Python environment for running this example. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37
pip install bigdl-orca[ray]
pip install torch==1.7.1 torchvision==0.8.2
```

### **Step 1: Init Orca Context**
```python
from bigdl.orca import init_orca_context, stop_orca_context

if cluster_mode == "local":  # For local machine
    init_orca_context(cores=4, memory="10g")
elif cluster_mode == "k8s":  # For K8s cluster
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1)
elif cluster_mode == "yarn":  # For Hadoop/YARN cluster
    init_orca_context(cluster_mode="yarn", cores=2, num_nodes=2, memory="10g", driver_memory="10g", driver_cores=1)
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](./../Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

### **Step 2: Define the Model**

You may define your model, loss and optimizer in the same way as in any standard (single node) PyTorch program.

```python
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

criterion = nn.NLLLoss()
```
After defining your model, you need to define a *Model Creator Function* that returns an instance of your model, and a *Optimizer Creator Function* that returns a PyTorch optimizer.

```python
def model_creator(config):
    model = LeNet()
    return model

def optim_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=0.001)
```

### **Step 3: Define Train Dataset**

You can define the dataset using a *Data Creator Function* that returns a PyTorch `DataLoader`. Orca also supports [Orca SparkXShards](../Overview/data-parallel-processing).

```python
import torch
from torchvision import datasets, transforms

torch.manual_seed(0)
batch_size = 320
test_batch_size = 320
dir = './dataset'

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
```

### **Step 4: Fit with Orca Estimator**

First, Create an Estimator

```python
from bigdl.orca.learn.pytorch import Estimator 
from bigdl.orca.learn.metrics import Accuracy

est = Estimator.from_torch(model=model_creator, optimizer=optim_creator, loss=criterion, metrics=[Accuracy()],
                           backend="torch_distributed")
```

Next, fit and evaluate using the Estimator

```python
est.fit(data=train_loader_creator, epochs=1, batch_size=batch_size)
result = est.evaluate(data=test_loader_creator, batch_size=test_batch_size)
for r in result:
    print(r, ":", result[r])
```

**Note:** You should call `stop_orca_context()` when your application finishes.
