# Enable AutoML for PyTorch

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/autoestimator_pytorch_lenet_mnist.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/autoestimator_pytorch_lenet_mnist.ipynb)

---

**In this guide we will describe how to enable automated hyper-parameter search for PyTorch using Orca `AutoEstimator`.**

### **Step 0: Prepare Environment**

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is needed to prepare the Python environment for running this example. Please refer to the [install guide](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-tuning.html#install) for more details.

```bash
conda create -n bigdl-orca-automl python=3.7 # bigdl-orca-automl is conda environment name, you can use any name you like.
conda activate bigdl-orca-automl
pip install bigdl-orca[automl]
pip install torch==1.8.1 torchvision==0.9.1
```

### **Step 1: Init Orca Context**
```python
from bigdl.orca import init_orca_context, stop_orca_context

if cluster_mode == "local":
    init_orca_context(cores=4, memory="2g", init_ray_on_spark=True) # run in local mode
elif cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4, init_ray_on_spark=True) # run on K8s cluster
elif cluster_mode == "yarn":
    init_orca_context(
      cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g", init_ray_on_spark=True, 
      driver_memory="10g", driver_cores=1) # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](./../Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

### **Step 2: Define the Model**

You may define your model, loss and optimizer in the same way as in any standard PyTorch program.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, fc1_hidden_size=500):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, fc1_hidden_size)
        self.fc2 = nn.Linear(fc1_hidden_size, 10)

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
After defining your model, you need to define a *Model Creator Function* that returns an instance of your model, and a *Optimizer Creator Function* that returns a PyTorch optimizer. Note that both the *Model Creator Function* and the *Optimizer Creator Function* should take `config` as input and get the hyper-parameter values from `config`.

```python
def model_creator(config):
    model = LeNet(fc1_hidden_size=config["fc1_hidden_size"])
    return model

def optim_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=config["lr"])
```

### **Step 3: Define Dataset**

You can define the train and validation datasets using *Data Creator Function* that takes `config` as input and returns a PyTorch `DataLoader`.

```python
import torch
from torchvision import datasets, transforms

torch.manual_seed(0)
dir = './dataset'
test_batch_size = 640

def train_loader_creator(config):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=config["batch_size"], shuffle=True)
    return train_loader

def test_loader_creator(config):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=False)
    return test_loader
```

### **Step 4: Define Search Space**
You should define a dictionary as your hyper-parameter search space.

The keys are hyper-parameter names which should be the same with those in your creators, and you can specify how you want to sample each hyper-parameter in the values of the search space. See [automl.hp](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-hp) for more details.

```python
from bigdl.orca.automl import hp

search_space = {
    "fc1_hidden_size": hp.choice([500, 600]),
    "lr": hp.choice([0.001, 0.003]),
    "batch_size": hp.choice([160, 320, 640]),
}
```

### **Step 5: Automatically Fit and Search with Orca AutoEstimator**

First, create an `AutoEstimator`. You can refer to [AutoEstimator API doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-auto-estimator) for more details.

```python
from bigdl.orca.automl.auto_estimator import AutoEstimator

auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                    optimizer=optim_creator,
                                    loss=criterion,
                                    logs_dir="/tmp/orca_automl_logs",
                                    resources_per_trial={"cpu": 2},
                                    name="lenet_mnist")
```

Next, use the `AutoEstimator` to fit and search for the best hyper-parameter set.

```python
auto_est.fit(data=train_loader_creator,
             validation_data=test_loader_creator,
             search_space=search_space,
             n_sampling=2,
             epochs=1,
             metric="accuracy")
```

Finally, you can get the best learned model and the best hyper-parameters.

```python
best_model = auto_est.get_best_model()
best_config = auto_est.get_best_config()
```

**Note:** You should call `stop_orca_context()` when your application finishes.
