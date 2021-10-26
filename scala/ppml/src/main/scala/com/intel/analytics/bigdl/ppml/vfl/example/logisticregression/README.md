# VFL Logistic Regression Example

This example show how to create an 2 trainer end-to-end VFL Logistic Regression application with BigDL PPML.

## Data
The data loading method is provided in code.

## Start FLServer
Before starting server, modify the config file, `ppml-conf.yaml`, this application has 2 trainer globally, so use following config.
```
worldSize: 2
```
Then start FLServer at server machine
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.FLServer
```

## Start Local Trainers
Change the config file to following.
```
clientTarget: FLServer_URL
```
And start the local Logistic Regression trainers at 2 training machines, with learning rate 0.01, batch size 4
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.example.logisticregression.VflLogisticRegression $dataPath $rowKeyName 0.01 4
```

The code in this exmaple will train the data and evaluate the training result.