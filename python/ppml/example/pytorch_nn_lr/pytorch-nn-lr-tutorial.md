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

### 2.1 Private Set Intersection
// TODO: add this section after Python version of PSI is done

### 2.2 Data Preprocessing
Since one party owns label data while another not, different operations should be done before training.

For example, in party 1:
```python
df_x = df_train.drop('Outcome', 1)
df_y = df_train['Outcome']

x = df_x.to_numpy(dtype="float32")
y = np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1)
```


### 2.3 Create Model
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
### 2.4 Create Estimator
Then, create Estimator and pass the arguments

```python
ppl = Estimator.from_torch(client_model=model,
                           client_id=client_id,
                           loss_fn=loss_fn,
                           optimizer_cls=torch.optim.SGD,
                           optimizer_args={'lr':1e-3},
                           target='localhost:8980',
                           server_model=server_model)
                           # if you want to upload server model from this estimator, pass server_model
                           # otherwise, ignore this argument

```

### 2.5 Training
Then call `fit` method to train

```python
response = ppl.fit(x, y)
```
### 2.7 Predict
```python
result = ppl.predict(x)
```

## 3 Run FGBoost
FL Server is required before running any federated applications. Check [Start FL Server]() section for details.
### 3.1 Start FL Server in SGX
#### 3.1.1 Start the container
Before running FL Server in SGX, please prepaer keys and start the BigDL PPML container first. Check  [3.1 BigDL PPML Hello World](https://github.com/intel-analytics/BigDL/tree/main/ppml#31-bigdl-ppml-hello-world) for details.
#### 3.1.2 Run start_fl_server.py
First, enter the container.
```bash
docker exec -it YOUR_CONTAINER bash
cd /ppml/trusted-big-data-ml
```
Then, write the following commands into a start-python-fl-server-sgx.sh.
```bash
#!/bin/bash
cd /ppml/trusted-big-data-ml
/graphene/Tools/argv_serializer bash -c " /opt/jdk8/bin/java     -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*'     -Xmx10g org.apache.spark.deploy.SparkSubmit     --master 'local[4]'   start-fl-server.py -p 8980 -c 2" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee fl-server.log
```
You can set port with `-p` or `--port` and set client number with `-c` or `--client-num` while the default settings are `port=8980` and `client-num=2`.
Then run the script,
```bash
bash start-python-fl-server-sgx.sh
```
or run it outside of the container.
```bash
docker exec -it YOUR_CONTAINER bash start-python-fl-server-sgx.sh
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
