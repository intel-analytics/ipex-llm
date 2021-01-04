# PyTorch Quickstart

---


**In this guide we will describe how to scale out PyTorch (v1.5+) programs using Orca in 4 simple steps.**

### **Step 0: Prepare Environment**

We recommend you to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../PythonUserGuide/install/) for more details.

**Note:** Conda environment is required to run on the distributed cluster, but not strictly necessary for running on the local machine.

```bash
conda create -n zoo python=3.7 # zoo is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo # install either version 0.9 or latest nightly build
pip install torch==1.7.1 torchvision==0.8.2
pip install six cloudpickle
pip install jep==3.9.0
```

**Note:** The original [source code](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/examples/orca/learn/pytorch/mnist/lenet_mnist.py) for the tutorial below only supports torch version >= 1.5.

### **Step 1: Init Orca Context**
```python
from zoo.orca import init_orca_context, stop_orca_context


if args.cluster_mode == "local":
    init_orca_context(cores=1, memory="2g")   # run in local mode
elif args.cluster_mode == "yarn":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(
    cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
    driver_memory="10g", driver_cores=1,
    conf={"spark.rpc.message.maxSize": "1024",
        "spark.task.maxFailures": "1",
        "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})   # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](./context) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when you run on Hadoop YARN cluster.

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
        
model = LeNet()
model.train()
criterion = nn.NLLLoss()
adam = torch.optim.Adam(model.parameters(), 0.001)
```

### **Step 3: Define Train Dataset**

You can define the dataset using standard [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html). Orca also supports a data creator function or [Orca SparkXShards](./data).

```python
import torch
from torchvision import datasets, transforms

torch.manual_seed(0)
dir='./dataset'
batch_size=64
test_batch_size=64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dir, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=False) 
```

### **Step 4: Fit with Orca Estimator**

First, Create an Estimator

```python
from zoo.orca.learn.pytorch import Estimator 

est = Estimator.from_torch(model=model, optimizer=adam, loss=criterion)
```

Next, fit and evaluate using the Estimator

```python
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch 

est.fit(data=train_loader, epochs=10, validation_data=test_loader,
        validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())

result = est.evaluate(data=test_loader, validation_methods=[Accuracy()])
for r in result:
    print(str(r))
```

**Note:** You should call `stop_orca_context()` when your application finishes.
