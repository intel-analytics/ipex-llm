
**In this guide we will describe how to scale out PyTorch (v1.5 or above) programs using Orca in 4 simple steps.**

<a target="_blank" href="https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>&nbsp; <a target="_blank" href="https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>

### **Step 0: Prepare Environment**

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is needed to prepare the Python environment for running this example. Please refer to the [install guide](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/python.html) for more details.

```bash
conda create -n zoo python=3.7 # zoo is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo # install either version 0.9 or latest nightly build
pip install torch==1.7.1 torchvision==0.8.2
pip install six cloudpickle
pip install jep==3.9.0
```

### **Step 1: Init Orca Context**
```python
from zoo.orca import init_orca_context, stop_orca_context


if args.cluster_mode == "local":
    init_orca_context(cores=1, memory="2g")   # run in local mode
elif args.cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(
    cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
    driver_memory="10g", driver_cores=1,
    conf={"spark.rpc.message.maxSize": "1024",
        "spark.task.maxFailures": "1",
        "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})   # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for more details.

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
from zoo.orca.learn.metrics import Accuracy 

est = Estimator.from_torch(model=model, optimizer=adam, loss=criterion, metrics=[Accuracy()])
```

Next, fit and evaluate using the Estimator

```python
from zoo.orca.learn.trigger import EveryEpoch 

est.fit(data=train_loader, epochs=10, validation_data=test_loader,
        checkpoint_trigger=EveryEpoch())

result = est.evaluate(data=test_loader)
for r in result:
    print(str(r))
```

**Note:** You should call `stop_orca_context()` when your application finishes.
