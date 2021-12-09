# PPML Demo

## Get Prepared
### Get jar ready
#### Build from source
```bash
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/scala
./make-dist.sh
```
#### Download pre-build
```bash
wget
```
### Config
If deploying PPML on cluster, need to overwrite config `./ppml-conf.yaml`. Default config (localhost:8980) would be used if no `ppml-conf.yaml` exists in the directory.
### Start FL Server
```bash
java -cp com.intel.analytics.bigdl.ppml.FLServer
```
## HFL Logistic Regression
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.hfl_logistic_regression.HflLogisticRegression -data/diabetes-hfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.hfl_logistic_regression.HflLogisticRegression -data/diabetes-hfl-2.csv
```
## VFL Logistic Regression
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.vfl_logistic_regression.VflLogisticRegression -data/diabetes-vfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.vfl_logistic_regression.VflLogisticRegression -data/diabetes-vfl-2.csv
```