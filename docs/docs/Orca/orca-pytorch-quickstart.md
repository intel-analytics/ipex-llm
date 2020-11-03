
**In this guide we’ll show you how to organize your PyTorch code into Orca in 3 steps.**

Scaling your Pytorch applications with Orca makes your code:

* Well-organized and flexible
* Easier to reproduce
* Able to perform distributed training without changing your model

### **Step 0: Prepare Environment**
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).

Download and install latest analytics-zoo whl by the following instructions [here](../PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip).  

**Note:** Conda environment is required to run on Yarn, but not strictly necessary for running on local.

```bash
conda create -n zoo python=3.7 # zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl 
pip install jep==3.9.0
conda install pytorch torchvision cpuonly -c pytorch # command for linux
conda install pytorch torchvision -c pytorch # command for macOS
```

### **Step 1: Init Orca Context**
```python
from zoo.orca import init_orca_context, stop_orca_context

# run in local mode
sc = init_orca_context(cores=1, memory="20g")

# run in yarn client mode
sc = init_orca_context(
    cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
    driver_memory="10g", driver_cores=1,
    conf={"spark.rpc.message.maxSize": "1024",
        "spark.task.maxFailures": "1",
        "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
```
**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir`. 

View [Orca Context](./context) for more details.

### **Step 2: Define PyTorch Model, Loss function and Optimizer**
```python
import torch.nn as nn
import torch.nn.functional as F
from bigdl.optim.optimizer import Adam

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
adam = Adam(1e-4)
```

### **Step 3: Fit with Orca PyTorch Estimator**  

1)  Define the data in whatever way you want. Orca just needs a [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html), a data creator function or [Orca SparkXShards](./data).
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

2)  Create an Estimator
```python
from zoo.orca.learn.pytorch import Estimator 

zoo_estimator = Estimator.from_torch(model=model, optimizer=adam, loss=criterion, backend="bigdl") 
```

3)  Fit with Estimator

```python
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch 

zoo_estimator.fit(data=train_loader, epochs=10, validation_data=test_loader,
                  validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch()) 
```

**Note:** You should call `stop_orca_context()` when your application finishes.
