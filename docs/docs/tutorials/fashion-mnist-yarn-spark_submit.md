In this tutorial, you will learn how to build, submit and execute a BigDL PyTorch example with `spark-submit`. 

In particular, we will show you:
1. How to prepare for the environment;
2. How to use BigDL orca estimator to conduct a PyTorch example;
3. How to submit the application to Yarn.


# Prepare Env
We need to first use conda to prepare the Python environment on the local machine where we submit our application. Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:

``` bash
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl

pip install bigdl
pip install conda-pack # use conda-pack to pack the conda environment to an archive
# Use conda or pip to install all the needed Python dependencies in the created conda environment.
```

Note: If the driver node in your cluster cannot install conda, you may install all dependency on a node which has conda envionment instead.

Check the Hadoop setup and configurations of our cluster. Make sure we properly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:

```bash
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
```

# Training the Fashion-Mnist Example

In this part, you will learn how to complete a BigDL PyTorch program with the following steps in order:
1. Prepare the Fashion-Mnist Dataset
2. Define a Convolutional Neural Network
3. Define a Loss and optimizer function
4. Initialize the OrcaContext
5. Create an Estimator for PyTorch
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

we can define the model just the same as using PyTorch.

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

An Orca program usually starts with the initialization of OrcaContext, we can easily specific the `runtime`(default is `spark`) and `cluster_mode` to create or get a SparkContext or RayContext with optimized configurations for BigDL performance.

To submit the BigDL program to Yarn with `spark_submit`, we just need to specific the `cluster_mode` to `spark-submit`. In this case, it's necessary for us to set the Spark configurations through command line options or the properties file.

```python
from bigdl.orca import init_orca_context, stop_orca_context

init_orca_context(cluster_mode="spark-submit")
```

## 5. Create the PyTorch Estimator

We could simply create a PyTorch estimator, the Orca Estimator will replicate the model on each node in the cluster, feed the data partition on each node to the local model replica, and synchronize model parameters using various backend technologies.

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

Note: The backend of Estimator must not be `bigdl` when we are running with `spark-submit`, it's better to use `bigdl` backend when we initializing OrcaContext with `yarn-client` mode and run the program with built-in function.

## 6. Training and Testing the Model

It's time to call `Estimator.fit` to train the network on training dataset, we can simply feed the input to the model and set times to loop over input. 

```python
stats = orca_estimator.fit(train_data_creator, epochs=epochs, batch_size=batch_size)
print("Train stats: {}".format(stats))
```
To evaluate the network on testing dataset, we can call `Estimator.evaluate` which will return a dictionary of metrics for the given data, including validation accuracy and loss.

```python
val_stats = orca_estimator.evaluate(validation_data_creator, batch_size=batch_size)
print("Validation stats: {}".format(val_stats))
```

# Run BigDL Program on Yarn with Spark-Submit

The BigDL supports submmiting application to Yarn cluster, we just need refer to the following steps in order:

1. Pack the conda environment to an archive.

```bash
conda pack -f -o environment.tar.gz
```

This archive file captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies, we can ship it together with scripts by using the `--archives` option, i.e. `--archives environment.tar.gz#environment`. It will be automatically unpacked on executors.

2. Submit with `spark-sumit-with-bigdl` script.

Now, let's submit our BigDL program to Yarn with `spark-sumit-with-bigdl` script. 

To load BigDL jars to Yarn Cluser, the script sets `spark.driver.extraClassPath` and `spark.executor.extraClassPath` to the location of jars files in BigDL package.

## Run BigDL on Yarn-Client Mode

For `yarn-client`, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment while the executors will use the Python interpreter in `environment.tar.gz`.

```bash
export PYSPARK_DRIVER_PYTHON='which python' # python location on driver
PYSPARK_PYTHON=environment/bin/python spark-submit-with-bigdl \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    fashion-mnist.py
```

## Run BigDL on Yarn-Cluster Mode

For `yarn-cluster` mode, the `PYSPARK_DRIVER_PYTHON` above should not be set, the Spark driver is running in a YARN container as well and thus both the driver and executors will use the Python interpreter in `environment.tar.gz` while the archive will be uploaded to remote resources like HDFS automatically. 

```bash
spark-submit-with-bigdl \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    fashion-mnist.py
```

In this script, we need to set the environment of the Spark driver and executor as the Python interpreter in the conda archive through `spark.yarn.appMasterEnv.PYSPARK_PYTHON` and `spark.executorEnv.PYSPARK_PYTHON` argument. 

## Run BigDL on Yarn-Client Mode when Driver cannot Install Conda
If your driver node cannot install conda, you could refer to the following steps, we will show you how to submit a BigDL program to yarn cluster with downloading BigDL package.

1. Install all the dependency files that BigDL required and pack the conda environment to an archive on the node which can install conda;
2. Send the packed conda archive to the driver node;
3. Download BigDL assembly packagefrom [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html) and set the unzipped file location as `${BIGDL_HOME}`.

Before submitting this application to Yarn, we need to prepare the environment configuration first.

```bash
export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf
export BIGDL_PY_ZIP=`find ${BIGDL_HOME}/python -name bigdl-spark_*-python-api.zip`
export PYSPARK_DRIVER_PYTHON='which python'
```

Then we can run the application on Yarn with the following `spark-submit` script:

```bash
PYSPARK_PYTHON=environment/bin/python ${SPARK_HOME}/bin/spark-submit \
    --properties-file ${BIGDL_CONF} \
    --py-files ${BIGDL_PY_ZIP} \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    fashion_mnist.py
```

In this script, we need to:
1. Specific the `--archives` argument to the location of the archive which was sent from the other node. 
2. Specific the `--properties-file` argument to override spark configuration by `${BIGDL_CONF}`.
3. Specific the `--py-files` argument as the `${BigDL_PY_ZIP}` file to for dependency libriaies.
4. Specific the  `spark.driver.extraClassPath` and `spark.executor.extraClassPath` argument to use the BIGDL jars files to prepend to the classpath of the driver and executors. 
