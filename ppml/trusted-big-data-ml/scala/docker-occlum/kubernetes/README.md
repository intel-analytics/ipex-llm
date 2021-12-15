# Spark 3.1.2 on K8S with Occlum

## Prerequisite

* Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)
* Prepare image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT`

1. Pull image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT
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

Prepare UCI dataset [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) and put this file in folder `/tmp`. You can change the path to iris.data via change mount path `data-exchange` in `executor.yaml`.
```bash
./run_spark_xgboost.sh
```

Parameters in run_spark_xgboost.sh:

* path_to_iris.data : String. 

    For example, yout host path to iris.data is `/tmp/iris.data` then this parameter in `run_spark_xgboost.sh` is `/host/data/iris.data`.
* num_threads : Int
* num_round : Int
* path_to_model_to_be_saved : String. 

    After training, you can find xgboost model in folder `/tmp/path_to_model_to_be_saved` if this parameter is `/host/data/xgboost_model_to_be_saved`

**note: make sure num_threads is larger than spark.task.cpus.**
