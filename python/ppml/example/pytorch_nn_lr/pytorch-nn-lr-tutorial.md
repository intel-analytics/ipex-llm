# Tutorial for Vertical Pytorch NN Logistic Regression
This example provides a step-by-step tutorial of running a Pytorch NN Logistic Regression task with 2 parties.
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


## 2. Build Pytorch NN Client Applications
This section introduces the details of the example code.

We use [Diabetes](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset. To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts. The split is taken by select every other column (code at [split script](scala/ppml/scripts/split_dataset.py)).

The code is available in projects, including [Client 1 code](fgboost_regression_party_1.py) and [Client 2 code](fgboost_regression_party_2.py). You could directly start two different terminals are run them respectively to start a federated learning, and the order of start does not matter. Following is the detailed step-by-step tutorial to introduce how the code works.

### 2.1 Initialize FL Context
We first need to initialize the FL Context by
```python
from bigdl.ppml.fl.nn.fl_context import init_fl_context
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

### 2.3 Data Preprocessing
Since one party owns label data while another not, different operations should be done before training.

For example, in party 1:
```python
df_x = df_train.drop('Outcome', 1)
df_y = df_train['Outcome']

x = df_x.to_numpy(dtype="float32")
y = np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1)
```


### 2.4 Create Model
We create the following model for both clients, but with different number of inputs 
```python
class LocalModel(nn.Module):
    def __init__(self, num_feature) -> None:
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        x = self.dense(x)
        return x

model = LocalModel(len(df_x.columns))
loss_fn = nn.BCELoss()
```
Besides, in party 1, we also define the server model and will upload it to FL Server later
```python
class ServerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.stack(x)
        x = torch.sum(x, dim=0) # above two act as interactive layer, CAddTable
        x = self.sigmoid(x)
        return x

server_model = ServerModel()
```
### 2.5 Create Estimator
Then, create Estimator and pass the arguments

```python
ppl = Estimator.from_torch(client_model=model,
                           loss_fn=loss_fn,
                           optimizer_cls=torch.optim.SGD,
                           optimizer_args={'lr':1e-4},
                           server_model=server_model,
                           server_model_path=/path/to/model/on/server,
                           client_model_path=/path/to/model/on/client)
```
Note that
* If you want to upload server model from this estimator, provide `server_model` argument.
* If you want server to automatically trigger model autosave, provide `server_model_path` with the path for server to save the model.
* If you want client to automatically trigger model autosave, provide `client_model_path` with the path for client to save the model.
We will also show how to use the saved model to resume training or predict in [2.8](#28-saveload)

### 2.6 Training
Then call `fit` method to train

```python
response = ppl.fit(x, y, epoch=5)
```

### 2.7 Predict
```python
result = ppl.predict(x)
```

### 2.8 Save/Load
In [2.5](#25-create-estimator) we provided the model paths while creating estimator. Thus, client model and server model would both be automatically saved.

You can also call save explicitly by
```python
torch.save(ppl.model, model_path) # save client model on local
ppl.save_server_model(server_model_path) # save server model on server
```
To start a new application to resume training or predict
```python
client_model = torch.load(model_path) # load client model first

# create estimator using the loaded client model
# the server_model_path should be consistant with the one in 2.4
# because server would load model from this path if model exists
ppl = Estimator.from_torch(client_model=model,
                           loss_fn=loss_fn,
                           optimizer_cls=torch.optim.SGD,
                           optimizer_args={'lr':1e-4},
                           server_model_path=/path/to/model/on/server,
                           client_model_path=/path/to/model/on/client)
ppl.load_server_model(server_model_path) # trigger model loading on server

# Then you can use the loaded model to resume training or predict
ppl.fit(x, y, epoch=5)
result = ppl.predict(x)
```
## 3 Run FGBoost
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
python pytorch_nn_lr_1.py
```
Start Client 2:
```bash
python pytorch_nn_lr_2.py
```
### 3.3 Get Result
The first 5 predict results are printed
```
[[2.3898797e-14]
 [1.8246214e-06]
 [6.7879863e-21]
 [1.2120417e-23]
 [0.0000000e+00]]
```
### 3.4 Incremental Training
Incremental training is supported, we just need to use the same configurations and start FL Server again.

In SGX container, start FL Server
```
./ppml/scripts/start-fl-server.sh 
```
For client applications, we change from creating model to directly loading. This is already implemented in example code, we just need to run client applications with an argument

```bash
# run following commands in 2 different terminals
python pytorch_nn_lr_1.py true
python pytorch_nn_lr_2.py true
```
The result based on new boosted trees are printed
```
[[1.8799074e-36]
 [1.7512805e-25]
 [4.6501680e-30]
 [1.4828590e-27]
 [0.0000000e+00]]
```
and you can see the loss continues to drop from the log of [Section 3.3](#33-get-results)
