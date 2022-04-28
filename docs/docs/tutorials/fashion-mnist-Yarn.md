In this tutorial, you will learn how to build, submit and execute a PyTorch example on Yarn using BigDL. 

In particular, we will show you:
1. How to prepare for the environment;
2. How to use BigDL orca estimator to conduct a PyTorch example;
3. How to run BigDL program on Yarn.

Let's get started!

### Prepare Env
We need to first use conda to prepare the Python environment on the local machine where we submit our application. Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:

``` bash
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl

pip install bigdl
# Use conda or pip to install all the needed Python dependencies in the created conda environment.
```

Check the Hadoop setup and configurations of our cluster. Make sure we properly set the environment variable HADOOP_CONF_DIR, which is needed to initialize Spark on YARN:

```bash
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
```

### Training the Fashion-Mnist Example

In this part, you will learn how to complete a BigDL PyTorch program with the following steps in order:
1. Prepare the Fashion-Mnist Dataset
2. Define a Convolutional Neural Network
3. Define a Loss and optimizer function
4. Initialize the OrcaContext
5. Initialize the PyTorch Estimator
6. Training and Test the Network

## 1. Prepare the Fashion-Mnist Dataset

For this tutorial, we will use the Fasion-Mnist dataset, it's easy to download and prepare using `torchvision`. The difference between PyTorch and BigDL is that the PyTorch estimator in BigDL need us to define a `data_creator` function which returns a PyTorch Dataloader instead.

```python
import torch
import torchvision
import torchvision.transforms as transforms

def train_data_creator(config={}, batch_size=4, download=True, data_dir='./data'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root=data_dir,
                                                 download=download,
                                                 train=True,
                                                 transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader

def validation_data_creator(config={}, batch_size=4, download=True, data_dir='./data'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return testloader
```

## 2. Define a Convolutional Neural Network

we can define a network just the same as using PyTorch.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

It's also necessary for us to define a `model_creator` function which returns the network for creating an estimator for PyTorch.

```python
def model_creator(config):
    model = Net()
    return model
```

## 3. Define a Loss and Optimizer Function

Let’s use a Classification Cross-Entropy loss and SGD with momentum, by the way, the optimizer should also be defined as a `optimizer_creator`.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

def optimizer_creator(model, config):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer
```

## 4. Initialize the OrcaContext

The interesting things start from now, we will see the magic power of BigDL.

An Orca program usually starts with the initialization of OrcaContext, we can easily specific the `runtime`(default is `spark`, `ray` is also a first-class backend) and `cluster_mode` to create or get a SparkContext or RayContext with optimized configurations for BigDL performance.

Here we could specific `cluster_mode` argument as `yarn-client` or `yarn-cluster` as following to start running BigDL program on Yarn. We only need to create a conda environment and install the python dependencies in that environment beforehand on the driver machine, these dependencies would be automatically packaged and distributed to the whole Yarn cluster.

```python
from bigdl.orca import init_orca_context, stop_orca_context

init_orca_context(cluster_mode="yarn-client") #or `yarn-cluster`
```

Note: For `yarn-client`, the Spark driver will run on the node where you start Python, while for `yarn-cluster` the Spark driver will run on a random node in the YARN cluster. So if you are running with `yarn-cluster`, you should change the application’s data loading from local file to a network file system (e.g. HDFS).

## 5. Create an Estimator for PyTorch

We could simply create an estimator for PyTorch now, the estimator will replicate the model on each node in the cluster, feed the data partition on each node to the local model replica, and synchronize model parameters using various backend technologies.

```python
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

orca_estimator = Estimator.from_torch(model=model_creator,
                                      optimizer=optimizer_creator,
                                      loss=criterion,
                                      metrics=[Accuracy()],# Orca validation methods for evaluate.
                                      model_dir=os.getcwd(),# The path to save model
                                      backend="spark")
```

The PyTorch Estimator supports backends include `spark`, `ray`, `bigdl` and `Horovod`, we could specific for `backend` argument as we need.

## 6. Training and Testing the Model

It's time to call `Estimator.fit` to train the network on the `training_data_creator` function, we can simply feed the input to the model and set times to loop over input.

```python
stats = orca_estimator.fit(train_data_creator, epochs=epochs, batch_size=batch_size)
print("Train stats: {}".format(stats))
```
To evaluate the network on testing dataset, we can call `Estimator.evaluate` which will return a dictionary of metrics for the given data, including validation accuracy and loss.

```python
val_stats = orca_estimator.evaluate(validation_data_creator, batch_size=batch_size)
print("Validation stats: {}".format(val_stats))
```

### Run BigDL Program on Yarn with built-in function

For now we have completed a PyTorch example with BigDL, we can directly run it as a normal python script:

```bash
python fashion-mnist.py --cluster_mode yarn-client --data_dir data
```

output:

When the `cluster_mode` is `yarn-cluster`, we could run as:

```bash
python fashion-mnist.py --cluster_mode yarn-cluster --data_dir hdfs://xxxx:port
```

### What is Next?

This is a directly way to run BigDL program on Yarn, we also provide a tutorial for users who want to run the BigDL program on Yarn with spark-submit.
