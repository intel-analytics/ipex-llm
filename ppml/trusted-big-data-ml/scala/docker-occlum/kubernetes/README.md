# Spark 3.1.2 on K8S with Occlum

## Prerequisite

* Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)
* Prepare image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT`

Pull image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT
```

If Dockerhub is not accessable, we can build docker image with Dockerfile and modify the path in the build-docker-image.sh firstly.

``` bash
cd ..
bash build-docker-image.sh
```

## Run Spark executor in Occlum:

### Run SparkPi example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_pi.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_pi.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_pi.sh
```

### Run Spark ML LogisticRegression example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_lr.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_lr.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_lr.sh
```

### Run Spark ML GradientBoostedTreeClassifier example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_gbt.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_gbt.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_gbt.sh
```

### Run Spark SQL SparkSQL example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_sql.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_sql.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_sql.sh
```

### Run Spark XGBoost example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_xgboost.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_xgboost.sh `
3. Prepare UCI dataset [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) and put this file in folder `/tmp`. You can change the path to iris.data via change mount path `data-exchange` in `executor.yaml`.
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
