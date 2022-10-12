### Data
We use [Census]() data in this example

To simulate the scenario of two parties, we use select different features of Census data.

The original data has 15 columns. In preprocessing, some new feature are created from the combinations of some existed columns.

* data of client 1: `age`, `education`, `occupation`, cross columns: `edu_occ`, `age_edu_occ`
* data of client 2: `relationship`, `workclass`, `marital_status`

### Start FLServer
Before starting server, modify the config file, `ppml-conf.yaml`, this application has 2 clients globally, so use following config.
```
worldSize: 2
```
Then start FLServer at server machine
```
java -cp bigdl-ppml-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.FLServer
```

## Start Local Trainers
Change the config file to following.
```
clientTarget: FLServer_URL
```
The port of server is provided in `ppml-conf.yaml` when server starts, default value `8980`. e.g. if you run the server and clients on same machine, `FLServer_URL` should be `localhost:8980`

And start the local Logistic Regression trainers at 2 training machines
```
java -cp bigdl-ppml-[version]-jar-with-all-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.logisticregression.VflLogisticRegression 
    --dataPath dataset/diabetes/diabetes-1.csv 
    --rowKeyName ID
    --learningRate 0.005
    --batchSize 4    
# change dataPath to diabetes-2.csv at client-2
```

The example will train the data and evaluate the training result.
