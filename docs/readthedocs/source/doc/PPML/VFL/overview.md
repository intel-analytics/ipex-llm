# Vertical Federated Learning
Vertical Federated Learning (VFL) is a federated machine learning case where multiple data sets share the same sample ID space but differ in feature space. 

VFL is supported in BigDL PPML. It allows users to train a federated machine learning model where data features are held by different parties. In BigDL PPML, the following VFL scenarios are supported.
* **Private Set Intersection**: To get data intersection of different VFL parties.
* **Neural Network Model**: To train common neural network model with Pytorch or Tensorflow backend across VFL parties.
* **FGBoost Model**: To train gradient boosted decision tree (GBDT) model across multiple VFL parties.

## Key Concepts
A **FL Server** is a gRPC server to handle requests from FL Client. A **FL Client** is a gRPC client to send requests to FL Server. These requests include:
* serialized model to use in training at FL Server
* some model related instance, e.g. loss function, optimizer
* the Tensor which FL Server and FL Client interact, e.g. predict output, label, gradient

A **FL Context** is a singleton holding a FL Client instance. By default, only one instance is held in a FL application. And the gRPC channel in this singleton instance could be reused in multiple algorithms.

## Quick Start
This section provides an quick start example for each supported VFL case.

### Single Party End-to-end Walkthrough
This example could be directly run once the dependencies installed.

Note that following code are mostly client code. In practical case, before client application starts, a FL Server is required to started in order to handle the requests from the clients. However, in order to make this local end-to-end application works fine, the python FL Server running code is also added before this example.
```python
from bigdl.ppml.fl.algorithms.psi import PSI
from bigdl.ppml.fl.fl_server import FLServer
from bigdl.ppml.fl import *

fl_server = FLServer()
fl_server.build()
fl_server.start()

init_fl_context()
psi = PSI()
salt = psi.get_salt()
key = ["k1", "k2"]
psi.upload_set(key, salt)
intersection = psi.download_intersection()
```

### Multi Party Examples
In following examples, the provided code are client code. Before run following example, start a FL Server first. Note that the client number argument need to be set when starting FL Server (set to 2 if there are 2 parties in the FL case). In order to see detail of how to start FL Server, please refer to [this]()

Once ready, for each FL party, we can start a following example application.
#### Private Set Intersection
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.psi import PSI

init_fl_context()
psi = PSI()
salt = psi.get_salt()
key = ["k1", "k2", "k3"] # the key list from different parties are usually different
psi.upload_set(key, salt)
intersection = psi.download_intersection()
```
For more detail about Private Set Intersection, please refer to [this]()
#### Neural Network Model
```python
# TODO: to add after Estimator style refactor
```
For more detail about Private Set Intersection, please refer to [this]()
#### FGBoost Model
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
from bigdl.ppml.fl.utils import init_fl_context
import numpy as np

init_fl_context()
fgboost_regression = FGBoostRegression()
x, y = np.ones([2, 3]), np.ones([2]) # the data from different parties are usualy different
fgboost_regression.fit(x, y)
result = fgboost_regression.predict(x)
```
For more detail about Private Set Intersection, please refer to [this]()

## Lifecycle

## Fault Tolerance
