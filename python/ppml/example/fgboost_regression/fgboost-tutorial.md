# Tutorial for Vertically Federated XGBoost
This example provides a step-by-step tutorial of running a FGBoost Regression task with 2 parties.
## 1. Key Concepts
### 1.1 FGBoost
**FGBoost** implements a Vertical Federated Learning algorithm for XGBoost. It allows multiple parties to run a federated gradient boosted decision tree application.
### 1.2 PSI
**PSI** stands for Private Set Intersection algorithm. It compute the intersection of data from different parties and return the common data IDs that all the parties holds so that parties know which data they could collaborate with.
### 1.3 SGX
**SGX** stands for [Software Guard Extensions](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html). It helps protect data in use via unique application isolation technology. The data of application running in SGX would be protected.

### 1.4 FL Server and Client
**FL Server** and **FL CLient** are parts of BigDL PPML using gRPC to communicate with each other.
* A centralized FL Server runs in SGX, aggregates intermediate results of clients, and return aggregated results (loss, gradients)
* A FL Client runs on each party, computes intermediate result on local data, and updates partial model based on aggregated results from the FL Server


## 2. Build FGBoost Client Applications
This section introduces the details of the example code.

We use [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset. To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts. The split is taken by select every other column (code at [split script](scala/ppml/scripts/split_dataset.py)).

The code is available in projects, including [Client 1 code](fgboost_regression_party_1.py) and [Client 2 code](fgboost_regression_party_2.py). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.

### 2.1 FL Context
First, initilize FL Context.

```python
# create a singleton context in this process, 
# all FL algorithms will use this context with client_id later
init_fl_context(id=client_id) 
```

### 2.2 Private Set Intersection
Then, read the data,

```python
df_train = pd.read_csv(data_path)
```

To get the data intersection which the 2 parties can do federated learning, we have to the Private Set Intersection (PSI) algorithm first.

```python
from bigdl.ppml.fl.algorithms.psi import PSI
df_train = pd.read_csv(data_path_train)
df_train['Id'] = df_train['Id'].astype(str)

df_test = pd.read_csv(data_path_test)
df_test['Id'] = df_test['Id'].astype(str)
psi = PSI()
intersection = psi.get_intersection(list(df_train['Id']))
df_train = df_train[df_train['Id'].isin(intersection)]
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

```python
# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=10)

# party 2 owns label, so pass the features and the labels
fgboost_regression.fit(df_x, df_y, feature_columns=df_x.columns, label_columns=['SalePrice'], num_round=10)
```
### 2.6 Save & Load
Save the client model and load it back

```python
fgboost_regression.save_model(model_path)
loaded = FGBoostRegression.load_model(model_path)
```
### 2.7 Prediction
```python
df_test = pd.read_csv(data_test)
result = loaded.predict(df_test, feature_columns=df_test.columns)
```
## 3 Run FGBoost
FL Server is required before running any federated applications. Check [Start FL Server]() section for details.

### 3.1 Start FL Server in SGX
// TODO: Add SGX section

Copy the configuration file `ppml/scripts/ppml-conf.yaml` to the current directory, and modify the config
```yaml
# the port server gRPC uses
serverPort: 8980

# the path server uses to save server model checkpoints
fgBoostServerModelPath: /tmp/fgboost_server_model

# the number of clients in this federated learning application
clientNum: 2
```
Note that we also set `fgBoostServerModelPath` which will be used in incremental training in [Section 3.4](#34-incremental-training)

Then start the FL Server
```
./ppml/scripts/start-fl-server.sh 
```
### 3.2 Start FGBoost Clients
Copy the configuration file `ppml/scripts/ppml-conf.yaml` to the current directory, and modify the config
```yaml
# the URL of server
clientTarget: localhost:8980
```
Run client applications

```bash
# run following commands in 2 different terminals
python fgboost_regression_party_1.py
python fgboost_regression_party_2.py
```
### 3.3 Get Results
The first 5 prediction results are printed
```
0-th result of FGBoost predict: 9.793853759765625
1-th result of FGBoost predict: 9.793853759765625
2-th result of FGBoost predict: 9.793853759765625
3-th result of FGBoost predict: 9.793853759765625
4-th result of FGBoost predict: 9.793853759765625
```
### 3.4 Incremental Training
Incremental training is supported, as long as `fgBoostServerModelPath` is specified in FL Server config, the server automatically saves the model checkpoints. Thus, we just need to use the same configurations and start FL Server again.

In SGX container, start FL Server
```
./ppml/scripts/start-fl-server.sh 
```
For client applications, we change from creating model to directly loading. This is already implemented in example code, we just need to run client applications with an argument

```bash
# run following commands in 2 different terminals
python fgboost_regression_party_1.py true
python fgboost_regression_party_2.py true
```
The result based on new boosted trees are printed
```
0-th result of FGBoost predict: 7.993928909301758
1-th result of FGBoost predict: 7.993928909301758
2-th result of FGBoost predict: 7.993928909301758
3-th result of FGBoost predict: 7.993928909301758
4-th result of FGBoost predict: 7.993928909301758
```
and you can see the loss continues to drop from the log of [Section 3.3](#33-get-results)