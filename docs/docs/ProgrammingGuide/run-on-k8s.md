The [Apache Spark on Kubernetes](https://apache-spark-on-k8s.github.io/userdocs/index.html) project enables
native support for submitting Spark application to a kubernetes cluster. As a deep learning library for Apache
Spark, BigDL can also run on Kubernetes by leveraging Spark on Kubernetes.

---
## **Prerequisites**

1. You need to have a running Kubernetes cluster that support Spark on Kubernetes. See [here](https://apache-spark-on-k8s.github.io/userdocs/running-on-kubernetes.html#prerequisites)

2. You need to spin up the **resource staging server** for dependency management. See [Dependency Management](https://apache-spark-on-k8s.github.io/userdocs/running-on-kubernetes.html#dependency-management) (This is optional if all your application dependencies are
packaged into your own custom docker image or resides in remote locations like HDFS. See [Dependency Management Without The Resource Staging Server](https://apache-spark-on-k8s.github.io/userdocs/running-on-kubernetes.html#dependency-management-without-the-resource-staging-server))

---
## **Docker images**

BigDL already published pre-built docker images that can be deployed into containers with pods.

The images are as follows:

|Component|Image|
|---|---|
|Spark Driver Image|intelanalytics/spark-driver:v2.2.0-kubernetes-0.5.0|
|Spark Executor Image|intelanalytics/spark-executor:v2.2.0-kubernetes-0.5.0|
|Spark Initialization Image|intelanalytics/spark-init:v2.2.0-kubernetes-0.5.0|
|PySpark Driver Image|intelanalytics/spark-driver-py:v2.2.0-kubernetes-0.5.0|
|PySpark Executor Image|intelanalytics/spark-executor-py:v2.2.0-kubernetes-0.5.0|

You may also build your own customized images. see instructions [here](https://github.com/intel-analytics/BigDL/tree/master/docker/BigDL).

---
## **Run BigDL examples**

Run BigDL on Kubernetes is quite easy once you meet the prerequisites above. For example,
to run the BigDL scala Lenet example:

```shell
SPARK_HOME=...
BIGDL_HOME=...
$SPARK_HOME/bin/spark-submit \
  --deploy-mode cluster \
  --class com.intel.analytics.bigdl.models.lenet.Train \
  --master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
  --kubernetes-namespace default \
  --conf spark.executor.instances=4 \
  --conf spark.app.name=bigdl-lenet \
  --conf spark.executor.cores=1 \
  --conf spark.cores.max=4 \
  --conf spark.kubernetes.driver.docker.image=intelanalytics/spark-driver:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.executor.docker.image=intelanalytics/spark-executor:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.initcontainer.docker.image=intelanalytics/spark-init:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.resourceStagingServer.uri=http://<address-of-any-cluster-node>:31000 \
  $BIGDL_HOME/lib/bigdl-0.4.0-SNAPSHOT-jar-with-dependencies.jar \
-f hdfs://master:9000/mnist \
-b 128 \
-e 2 \
--checkpoint /tmp
```

To run python lenet example:

```shell
SPARK_HOME=...
BIGDL_HOME=...
$SPARK_HOME/bin/spark-submit \
  --deploy-mode cluster \
  --master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
  --kubernetes-namespace default \
  --jars $BIGDL_HOME/lib/bigdl-0.4.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files $BIGDL_HOME/lib/bigdl-0.4.0-SNAPSHOT-python-api.zip \
  --conf spark.executor.instances=4 \
  --conf spark.app.name=bigdl-1 \
  --conf spark.executor.cores=1 \
  --conf spark.cores.max=4 \
  --conf spark.kubernetes.driver.docker.image=intelanalytics/spark-driver-py:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.executor.docker.image=intelanalytics/spark-executor-py:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.initcontainer.docker.image=intelanalytics/spark-init:v2.2.0-kubernetes-0.5.0-ubuntu-14.04 \
  --conf spark.kubernetes.resourceStagingServer.uri=http://<address-of-any-cluster-node>:31000 \
  bigdl/models/lenet/lenet5.py \
  --action train \
  --dataPath /tmp/mnist \
  -n 2
```