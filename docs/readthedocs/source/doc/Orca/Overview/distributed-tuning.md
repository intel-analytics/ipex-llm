# Distributed Hyper-Parameter Tuning

---

**Orca `AutoEstimator` provides similar APIs as Orca `Estimator` for distributed hyper-parameter tuning.** 

### **Install**
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment.
```bash
conda create -n bigdl-orca-automl python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate bigdl-orca-automl
pip install bigdl-orca[automl]
````
You can install the latest release version of BigDL Orca as follows:
```bash
pip install --pre --upgrade bigdl-orca[automl]
```
Note that with extra key of [automl], `pip` will automatically install the additional dependencies for distributed hyper-parameter tuning,
including `ray[tune]==1.2.0`, `psutil`, `aiohttp==3.7.0`, `aioredis==1.1.0`, `setproctitle`, `hiredis==1.1.0`, `async-timeout==3.0.1`, `xgboost`.

To use [Pytorch Estimator](#pytorch-autoestimator), you need to install Pytorch with `pip install torch==1.8.1`.
To use [TensorFlow/Keras AutoEstimator](#tensorflow-keras-autoestimator), you need to install Tensorflow with `pip install tensorflow==1.15.0`.


### **1. AutoEstimator**

To perform distributed hyper-parameter tuning, user can first create an Orca `AutoEstimator` from standard TensorFlow Keras or PyTorch model, and then call `AutoEstimator.fit`.

Under the hood, the Orca `AutoEstimator` generates different trials and schedules them on each mode in the cluster. Each trial runs a different combination of hyper parameters, sampled from the user-desired hyper-parameter space.
HDFS is used to save temporary results of each trial and all the results will be finally transferred to driver for further analysis. 

### **2. Pytorch AutoEstimator**

User could pass *Creator Function*s, including *Data Creator Function*, *Model Creator Function* and *Optimizer Creator Function* to `AutoEstimator` for training. 

The *Creator Function*s should take a parameter of `config` as input and get the hyper-parameter values from `config` to enable hyper parameter search.

#### **2.1 Data Creator Function**
You can define the train and validation datasets using *Data Creator Function*. The *Data Creator Function* takes `config` as input and returns a `torch.utils.data.DataLoader` object, as shown below.
```python
# "batch_size" is the hyper-parameter to be tuned.
def train_loader_creator(config):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=config["batch_size"], shuffle=True)
    return train_loader
```
The input data for Pytorch `AutoEstimator` can be a *Data Creator Function* or a tuple of numpy ndarrays in the form of (x, y), where x is training input data and y is training target data.

#### **2.2 Model Creator Function**
*Model Creator Function* also takes `config` as input and returns a `torch.nn.Module` object, as shown below.

```python
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self, fc1_hidden_size=500):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, fc1_hidden_size)
        self.fc2 = nn.Linear(fc1_hidden_size, 10)
    
    def forward(self, x):
        pass

def model_creator(config):
    # "fc1_hidden_size" is the hyper-parameter to be tuned.
    model = LeNet(fc1_hidden_size=config["fc1_hidden_size"])
    return model
```

#### **2.3 Optimizer Creator Function**
*Optimizer Creator Function* takes `model` and `config` as input, and returns a `torch.optim.Optimizer` object. 
```python
import torch
def optim_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=config["lr"])
```

Note that the `optimizer` argument in Pytorch `AutoEstimator` constructor could be a *Optimizer Creator Function* or a string, which is the name of Pytorch Optimizer. The above *Optimizer Creator Function* has the same functionality with "Adam".

#### **2.4 Create and Fit Pytorch AutoEstimator**
User could create a Pytorch `AutoEstimator` as below.
```python
from zoo.orca.automl.auto_estimator import AutoEstimator

auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                    optimizer=optim_creator,
                                    loss=nn.NLLLoss(),
                                    logs_dir="/tmp/zoo_automl_logs",
                                    resources_per_trial={"cpu": 2},
                                    name="lenet_mnist")
```
Then user can perform distributed hyper-parameter tuning as follows. For more details about the `search_space` argument, view the *search space and search algorithms* [page](#search-space-and-search-algorithms).
```python
auto_est.fit(data=train_loader_creator,
             validation_data=test_loader_creator,
             search_space=search_space,
             n_sampling=2,
             epochs=1,
             metric="accuracy")
```
Finally, user can get the best learned model and the best hyper-parameters for further deployment.
```python
best_model = auto_est.get_best_model() # a `torch.nn.Module` object
best_config = auto_est.get_best_config() # a dictionary of hyper-parameter names and values.
```
View the related [Python API doc](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-auto-estimator) for more details.

### **3. TensorFlow/Keras AutoEstimator**
Users can create an `AutoEstimator` for TensorFlow Keras from a `tf.keras` model (using a *Model Creator Function*). For example:

```python
def model_creator(config):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(config["hidden_size"],
                                                              input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.SGD(config["lr"]),
                  metrics=["mse"])
    return model

auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                    logs_dir="/tmp/zoo_automl_logs",
                                    resources_per_trial={"cpu": 2},
                                    name="auto_keras")
```

Then user can perform distributed hyper-parameter tuning as follows. For more details about `search_space`, view the *search space and search algorithms* [page](#search-space-and-search-algorithms).
```python
auto_est.fit(data=train_data,
             validation_data=val_data,
             search_space=search_space,
             n_sampling=2,
             epochs=1,
             metric="accuracy")
```
The `data` and `validation_data` in `fit` method can only be a tuple of numpy ndarrays. We haven't support *Data Create Function* now. The numpy ndarray should also be in the form of (x, y), where x is training input data and y is training target data.

Finally, user can get the best learned model and the best hyper-parameters for further deployment.
```python
best_model = auto_est.get_best_model() # a `torch.nn.Module` object
best_config = auto_est.get_best_config() # a dictionary of hyper-parameter names and values.
```
View the related [Python API doc](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-auto-estimator) for more details.

### **4. Search Space and Search Algorithms**
For Hyper-parameter Optimization, user should define the search space of various hyper-parameter values for neural network training, as well as how to search through the chosen hyper-parameter space.

#### **4.1 Basic Search Algorithms**

For basic search algorithms like **Grid Search** and **Random Search**, we provide several sampling functions with `automl.hp`. See [API doc](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-hp) for more details.

`AutoEstimator` requires a dictionary for the `search_space` argument in `fit`.
In the dictionary, the keys are the hyper-parameter names, and the values specify how to sample the search spaces for the hyper-parameters.

```python
from zoo.orca.automl import hp

search_space = {
    "fc1_hidden_size": hp.grid_search([500, 600]),
    "lr": hp.loguniform(0.001, 0.1),
    "batch_size": hp.choice([160, 320, 640]),
}
```

#### **4.2 Advanced Search Algorithms**
Beside grid search and random search, user could also choose to use some advanced hyper-parameter optimization methods, 
such as [Ax](https://ax.dev/), [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization), [Scikit-Optimize](https://scikit-optimize.github.io), etc. We supported all *Search Algorithms* in [Ray Tune](https://docs.ray.io/en/master/index.html). View the [Ray Tune Search Algorithms](https://docs.ray.io/en/master/tune/api_docs/suggestion.html) for more details.
Note that you should install the dependency for your search algorithm manually.

Take bayesian optimization as an instance. You need to first install the dependency with

```bash
pip install bayesian-optimization
```

And pass the search algorithm name to `search_alg` in `AutoEstimator.fit`.
```python
from zoo.orca.automl import hp

search_space = {
    "width": hp.uniform(0, 20),
    "height": hp.uniform(-100, 100)
}

auto_estimator.fit(
    data,
    search_space=search_space,
    metric="mean_loss",
    mode="min",
    search_alg="bayesopt",
)
```
See [API Doc](https://analytics-zoo.readthedocs.io/en/latest/doc/PythonAPI/AutoML/automl.html#orca-automl-auto-estimator) for more details.

### **5. Scheduler**
*Scheduler* can stop/pause/tweak the hyper-parameters of running trials, making the hyper-parameter tuning process much efficient.

We support all *Schedulers* in [Ray Tune](https://docs.ray.io/en/master/index.html). See [Ray Tune Schedulers](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#schedulers-ref) for more details.

User can pass the *Scheduler* name to `scheduler` in `AutoEstimator.fit`. The *Scheduler* names supported are "fifo", "hyperband", "async_hyperband", "median_stopping_rule", "hb_bohb", "pbt", "pbt_replay".
The default `scheduler` is "fifo", which just runs trials in submission order.

See examples below about how to use *Scheduler* in `AutoEstimator`. 
```python
scheduler_params = dict(
            max_t=50,
            grace_period=1,
            reduction_factor=3,
            brackets=3,
        )

auto_estimator.fit(
    data,
    search_space=search_space,
    metric="mean_loss",
    mode="min",
    search_alg="skopt",
    scheduler = "AsyncHyperBand",
    scheduler_params=scheduler_params
)
```
*Scheduler* shares the same parameters as ray tune schedulers.
And `scheduler_params` are extra parameters for `scheduler` other than `metric` and `mode`.
