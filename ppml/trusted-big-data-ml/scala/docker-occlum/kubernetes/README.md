# Spark 3.1.2 on K8S with Occlum

## Prerequisite

* Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)
* Prepare image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.1.0-SNAPSHOT`

1. Pull image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.1.0-SNAPSHOT
```

If Dockerhub is not accessable, we can build docker image with Dockerfile and modify the path in the build-docker-image.sh firstly.

``` bash
cd ..
bash build-docker-image.sh
```

2. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_pi.sh`.
3. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_pi.sh `
4. Modify `executor.yaml` for your need

## Run Spark executor in Occlum:

### Run SparkPi example

```bash
./run_spark_pi.sh
```

### Run Spark ML LogisticRegression example

```bash
./run_spark_lr.sh
```

### Run Spark ML GradientBoostedTreeClassifier example

```bash
./run_spark_gbt.sh
```

### Run Spark SQL SparkSQL example

```bash
./run_spark_sql.sh
```

### Run Spark XGBoost example

#### UCI dataset [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

Prepare UCI dataset `iris.data` and put this file in folder `/tmp/xgboost_data`. 
You can change the path to iris.data via change mount path `data-exchange` in `executor.yaml`.
Then:
```bash
./run_spark_xgboost.sh
```
Parameters:

* path_to_iris.data : String.

  For example, yout host path to iris.data is `/tmp/xgboost_data/iris.data` then this parameter in `run_spark_xgboost.sh` is `/host/data/xgboost_data`.
* num_threads : Int
* num_round : Int
* path_to_model_to_be_saved : String.

  After training, you can find xgboost model in folder `/tmp/path_to_model_to_be_saved`.


#### Criteo 1TB Click Logs [dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)

Split 50G data from this dataset and put it into `/tmp/xgboost_data`. 
Then change the `class` in [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/kubernetes/run_spark_xgboost.sh#L7) to
`com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExampleOnCriteoClickLogsDataset`.

Add these configurations to [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/kubernetes/run_spark_xgboost.sh):
```bash
    --conf spark.driver.extraClassPath=local:///opt/spark/jars/* \
    --conf spark.executor.extraClassPath=local:///opt/spark/jars/* \
    --conf spark.cores.max=64 \
    --conf spark.kubernetes.driverEnv.DRIVER_MEMORY=10g \
    --conf spark.kubernetes.driverEnv.SGX_MEM_SIZE="40GB" \
    --conf spark.kubernetes.driverEnv.META_SPACE=1024m \
    --conf spark.kubernetes.driverEnv.SGX_HEAP="10GB" \
    --conf spark.kubernetes.driverEnv.SGX_KERNEL_HEAP="4GB" \
    --conf spark.executorEnv.SGX_MEM_SIZE="178GB" \
    --conf spark.executorEnv.SGX_KERNEL_HEAP="4GB" \
    --conf spark.executorEnv.SGX_HEAP="150GB" \
    --executor-cores 32 \
    --executor-memory 10g \
    --driver-memory 10g
```
Change the `parameters` to:
```commandline
/host/data/xgboost_data /host/data/xgboost_criteo_model 32 100 10
```
Then:
```bash
./run_spark_xgboost.sh
```
Parameters:

* path_to_Criteo_data : String. 

    For example, yout host path to Criteo dateset is `/tmp/xgboost_data/criteo` then this parameter in `run_spark_xgboost.sh` is `/host/data/xgboost_data`.
* path_to_model_to_be_saved : String.

    After training, you can find xgboost model in folder `/tmp/path_to_model_to_be_saved`.

* num_threads : Int
* num_round : Int
* max_depth: Int. Tree max depth.

**note: make sure num_threads is larger than spark.task.cpus.**

#### Source code
You can find source code [here](https://github.com/intel-analytics/BigDL/tree/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/xgboost).

The console output looks like:
```
[INFO] [02/10/2022 14:57:04.244] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [0]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:07.296] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [1]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:10.071] [RabitTracker-akka.actor.default-dispatcher-7] [akka://RabitTracker/user/Handler] [2]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
```




### Run Spark TPC-H example

Modify the following configuration in `executor.yaml`.

```yaml
imagePullPolicy: Always

env:
- name: SGX_THREAD
  value: "256"
- name: SGX_HEAP
  value: "2GB"
- name: SGX_KERNEL_HEAP
  value: "2GB"
- name: SGX_MMAP
  value: "16GB"
```

Then run the script.

```bash
./run_spark_tpch.sh
```

