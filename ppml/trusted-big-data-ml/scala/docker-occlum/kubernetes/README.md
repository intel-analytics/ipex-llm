# Spark 3.0.0 on K8S with Occlum

## Pre-prerequisites

* Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)
* Prepare image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-k8s:0.14-SNAPSHOT`

Pull from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-k8s:0.14-SNAPSHOT
```

If Dockerhub is not accessable, we can build docker image with Dockerfile and modify the path in the build-docker-image.sh firstly.

``` bash
bash build-docker-image.sh
```

## Run Spark executor in Occlum:

1. Download [Spark 3.0.0](https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz), and setup `SPARK_HOME`. Or set `SPARK_HOME` in `run_spark_pi.sh`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_spark_pi.sh `
3. Modify `executor.yaml` for your need

```bash
./run_spark_pi.sh
```
