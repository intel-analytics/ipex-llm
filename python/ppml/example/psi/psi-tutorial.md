# Tutorial for Private Set Intersection
This example provides a step-by-step tutorial of running a PSI (Private Set Intersection) task with 2 parties.
## 1. Key Concepts
### 1.1 PSI
**PSI** stands for Private Set Intersection algorithm. It compute the intersection of data from different parties and return the result so that parties know which data they could collaborate with.
### 1.2 SGX
**SGX** stands for [Software Guard Extensions](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html). It helps protect data in use via unique application isolation technology. The data of application running in SGX would be protected.

### 1.3 FL Server and Client
A **FL Server** is a gRPC server to handle requests from FL Client. A **FL Client** is a gRPC client to send requests to FL Server. These requests include:
* serialized model to use in training at FL Server
* some model related instance, e.g. loss function, optimizer
* the Tensor which FL Server and FL Client interact, e.g. predict output, label, gradient


## 2. Build PSI Client Applications
This section introduces the details of the example code.

We use [Diabetes](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset. To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts. The split is taken by select every other column (code at [split script](scala/ppml/scripts/split_dataset.py)).

The code is available in projects, including [Client 1 code](fgboost_regression_party_1.py) and [Client 2 code](fgboost_regression_party_2.py). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.

### 2.1 Initialize FL Context
We first need to initialize the FL Context by
```python
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI
init_fl_context(client_id, target)
```
The target is the URL of FL Server and is `localhost:8980` by default.
### 2.2 Private Set Intersection
Then get the intersection of datasets across parties by Private Set Intersection algorithm.
```python
df_train['ID'] = df_train['ID'].astype(str)
psi = PSI()
intersection = psi.get_intersection(list(df_train['ID']))
df_train = df_train[df_train['ID'].isin(intersection)]
```

## 3 Run PSI Application
FL Server is required before running any federated applications. Check [Start FL Server]() section for details.
### 3.1 Start FL Server in SGX

#### 3.1.1 Start the container
Before running FL Server in SGX, please prepare keys and start the BigDL PPML container first. Check  [3.1 BigDL PPML Hello World](https://github.com/intel-analytics/BigDL/tree/main/ppml#31-bigdl-ppml-hello-world) for details.
#### 3.1.2 Run FL Server in SGX
You can run FL Server in SGX with the following command:
```bash
docker exec -it YOUR_DOCKER bash /ppml/trusted-big-data-ml/work/start-scripts/start-python-fl-server-sgx.sh -p 8980 -c 2
```
You can set port with `-p` and set client number with `-c`  while the default settings are `port=8980` and `client-num=2`.
### 3.2 Start FGBoost Clients
Modify the config file `ppml-conf.yaml`
```yaml
# the URL of server
clientTarget: localhost:8980
```
Run client applications

Start Client 1:
```bash
python psi_1.py
```
Start Client 2:
```bash
python psi_2.py
```
### 3.3 Get Result
The intersected results are printed
```
Intersection completed, size 768
['430.0', '254.0', '232.0', '727.0', '166.0']
```