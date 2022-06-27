# Tutorial for Vertically Federated XGBoost
This example provides a step-by-step tutorial of running a FGBoost Regression task with 2 parties.
## 1. Key Concepts
### 1.1 FGBoost 
**FGBoost** stands for Federated Gradient Boosted Tree algorithm. It allows multiple parties to run a federated gradient boosted decision tree application.
### 1.2 PSI
**PSI** stands for Private Set Intersection algorithm. It compute the intersection of data from different parties and return the result so that parties know which data they could collaborate with.
### 1.3 SGX
**SGX** stands for [Software Guard Extensions](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html). It helps protect data in use via unique application isolation technology. The data of application running in SGX would be protected.

### 1.4 FL Server and Client
A **FL Server** is a gRPC server to handle requests from FL Client. A **FL Client** is a gRPC client to send requests to FL Server. These requests include:
* serialized model to use in training at FL Server
* some model related instance, e.g. loss function, optimizer
* the Tensor which FL Server and FL Client interact, e.g. predict output, label, gradient


## 2. Build FGBoost Client Applications
This section introduces the details of the example code.

We use [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset. To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts. The split is taken by select every other column (code at [split script](scala/ppml/scripts/split_dataset.py)).

The code is available in projects, including [Client 1 code](fgboost_regression_party_1.py) and [Client 2 code](fgboost_regression_party_2.py). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.

### 2.1 FL Context
First, import the package and initilize FL Context.
```python
from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
```

### 2.2 Private Set Intersection
Then, read the data,

Party 1:
```python
df_train = pd.read_csv('house-prices-train-1.csv')
```

Party 2:
```python
df_train = pd.read_csv('house-prices-train-2.csv')
```

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
### 2.3 Data Preprocessing
We provide a `preprocess` method in the code, including normalization and one-hot. The output is a DataFrame with all numerical features. You can check the detail in the [code]()

```python
df_train = preprocess(df_train)
```
### 2.4 FGBoost Instance
Create the FGBoost instance `FGBoostRegression` with default parameters 
```
fgboost_regression = FGBoostRegression()
```
### 2.5 Training
Then call `fit` method to train

Party 1:
```python
# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)
```
Party 2:
```python
# party 2 owns label, so pass the features and the labels
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=100)
```
### 2.6 Save & Load
Save the model and load it back

Party 1:
```python
fgboost_regression.save_model('/tmp/fgboost_model_1.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')
```
Party 2:
```python
fgboost_regression.save_model('/tmp/fgboost_model_2.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_2.json')
```
### 2.7 Prediction
Party 1:
```python
df_test = pd.read_csv('house-prices-test-1')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```

Party 2:
```python
df_test = pd.read_csv('house-prices-test-2')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)
```
## 3 Run FGBoost
FL Server is required before running any federated applications. Check [Start FL Server]() section for details.
### 3.1 Start FL Server in SGX
To run server in SGX, pull image from dockerhub
```
docker pull intelanalytics/bigdl-ppml-trusted-fl-graphene:2.1.0-SNAPSHOT
```
prepare the enclave key
```
openssl genrsa -3 -out enclave-key.pem 3072
```
// TODO: CHECK

Modify the config file `ppml-conf.yaml`
```yaml
# the port server gRPC uses
serverPort: 8980

# the number of clients in this federated learning application
clientNum: 2
```
Then start the FL Server
```
./ppml/scripts/start-fl-server.sh 
```
### 3.2 Start FGBoost Clients
Modify the config file `ppml-conf.yaml`
```yaml
# the URL of server
clientTarget: localhost:8980
```
Run client applications

Start Client 1:
```bash
python fgboost_regression_party_1.py
```
Start Client 2:
```bash
python fgboost_regression_party_2.py
```
### 3.3 Get Results
The first 5 batches of results are printed
```
0-th result of FGBoost predict: [[9.74606]
 [9.74606]
 [9.74606]
 [9.74606]]
1-th result of FGBoost predict: [[9.74606]
 [9.74606]
 [9.74606]
 [9.74606]]
2-th result of FGBoost predict: [[9.74606]
 [9.74606]
 [9.74606]
 [9.74606]]
3-th result of FGBoost predict: [[9.74606]
 [9.74606]
 [9.74606]
 [9.74606]]
4-th result of FGBoost predict: [[9.74606]
 [9.74606]
 [9.74606]
 [9.74606]]

```