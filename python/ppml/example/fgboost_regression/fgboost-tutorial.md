# FGBoost Regression Tutorial
This example provides a step-by-step tutorial of running a FGBoost Regression task with 2 parties.
## Key Concepts
A **FL Server** is a gRPC server to handle requests from FL Client. A **FL Client** is a gRPC client to send requests to FL Server. These requests include:
* serialized model to use in training at FL Server
* some model related instance, e.g. loss function, optimizer
* the Tensor which FL Server and FL Client interact, e.g. predict output, label, gradient

A **FL Context** is a singleton holding a FL Client instance. By default, only one instance is held in a FL application. And the gRPC channel in this singleton instance could be reused in multiple algorithms.

A **FGBoost Instance** is an algorithm instance running federated gradient boosted tree algorithm, it does local training and use **FL Context** to communicate with **FL Server**. 

## Write Client Code
This section introduces the details of the example code.

We use [House Prices]() dataset. To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts. The split is taken by select every other column (code at [split script]()).

The code is available in projects, including [Client 1 code]() and [Client 2 code](). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.
### Config
Modify the config file `ppml-conf.yaml`
```yaml
# the URL of server
clientTarget: localhost:8980
```
### Import and Initialize Context
First, import the package and initilize FL Context.
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
```

### Prepare Data
Then, read the data, preprocess and split it into feature and label if necessary,

Party 1:
```python
df_train = pd.read_csv('house-prices-train-1.csv')
```

Party 2:
```python
df_train = pd.read_csv('house-prices-train-2.csv')
```
#### Private Set Intersection
To get the data intersection which the 2 parties can do federated learning, we have to the Private Set Intersection (PSI) algorithm first.
```python
from bigdl.ppml.fl.algorithms.psi import PSI
ids = df_train['Id']
psi = PSI()
salt = psi.get_salt()
psi.upload_set(ids, salt)
intersection = psi.download_intersection()
df_train = df_train.ix(intersection) # select the intersection part of training data
```
#### Preprocess Data
We provide a `preprocess` method in the code, including normalization and one-hot. The output is a DataFrame with all numerical features. You can check the detail in the [code]()

```python
df_train = preprocess(df_train)
```
### Create Model and Train
Create the FGBoost instance `FGBoostRegression` with default parameters and train
Party 1:
```python
fgboost_regression = FGBoostRegression()
# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)
```
Party 2:
```
fgboost_regression = FGBoostRegression()
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=100)
```


## Start FLServer
FL Server is required before running any federated applications. Check [Start FL Server]() section for details.
### Run in SGX
To run server in SGX, pull image from dockerhub
```
docker pull intelanalytics/bigdl-ppml-trusted-fl-graphene:2.1.0-SNAPSHOT
```
prepare the enclave key
```
openssl genrsa -3 -out enclave-key.pem 3072
```
// TODO: CHECK
### Configuration
Modify the config file `ppml-conf.yaml`
```yaml
# the port server gRPC uses
serverPort: 8980

# the number of clients in this federated learning application
clientNum: 2
```
### Run Start Script
Then,
```
./ppml/scripts/start-fl-server.sh 
```
## Start Federated Client Applications
To run clients in SGX, follow the instructions [above]().

### Run Federated Applications
Start Client 1:
```bash
python fgboost_regression_party_1.py
```
Start Client 2:
```bash
python fgboost_regression_party_2.py
```
