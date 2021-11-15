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

### Run Spark.pi example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_pi.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_pi.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_pi.sh
```

### Run Spark.LogisticRegression example

1. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_lr.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_lr.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_lr.sh
```

