There are two ways to submit PPML jobs:
* use [PPML CLI](#ppml-cli) to submit jobs manually
* use [helm chart](#helm-chart) to submit jobs automatically

## PPML CLI
### Description
The PPML Command Line Interface is a unified tool to submit ppml spark jobs on a cluster.

### Synopsis
Once a user application is bundled, it can be launched using the [bigdl-ppml-submit.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/bigdl-ppml-submit.sh) script. This script takes care of setting up sgx configuration and cluster & deploy mode, running PPML jobs in secure environment:
```
./bigdl-ppml-submit.sh [options] <application-jar> [application-arguments]
```
### Options
* The following parameters enable spark executor running on SGX. Check the [recommended configuration of sgx](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#1-bigdl-ppml-sgx-related-configurations).

    `--sgx-enabled` **true** -> enable spark executor running on sgx, **false** -> native on k8s without SGX. The default value is false. Once --sgx-enabled is set as true, you should also set other sgx-related options (--sgx-log-level, --sgx-driver-memory, --sgx-driver-jvm-memory, --sgx-executor-memory, --sgx-executor-jvm-memory) otherwise PPML CLI will throw an error.

    `--sgx-driver-jvm-memory` Set the sgx driver jvm memory, recommended setting is less than half of driver epc memory, in the same format as JVM memory strings with a size unit suffix ("k", "m", "g" or "t") (e.g. 512m, 2g).

    `--sgx-executor-jvm-memory` Set the sgx executor jvm memory, recommended setting is less than half of executor epc memory, in the same format as JVM memory strings with a size unit suffix ("k", "m", "g" or "t") (e.g. 512m, 2g).

* Except for the above sgx options, other options are exactly the same as [spark properties](https://spark.apache.org/docs/latest/configuration.html#available-properties)

    `--master` The master URL for the cluster (e.g. spark://23.195.26.187:7077)

    `--deploy-mode` Whether to deploy your driver on the worker nodes (cluster) or locally as an external client (client) (default: client)

    `--driver-memory` Amount of memory to use for the driver process, in the same format as JVM memory strings with a size unit suffix ("k", "m", "g" or "t") (e.g. 512m, 2g).

    `--driver-cores` Number of cores to use for the driver process, only in cluster mode.

    `--executor-memory` Amount of memory to use per executor process, in the same format as JVM memory strings with a size unit suffix ("k", "m", "g" or "t") (e.g. 512m, 2g).

    `--executor-cores` The number of cores to use on each executor.

    `--num-executors` The initial number of executors.

    `--name` The Spark application name is used by default to name the Kubernetes resources created like drivers and executors.

    `--verbose` Print out fine-grained debugging information

    `--class` The entry point for your application (e.g. org.apache.spark.examples.SparkPi)

    `application-jar`: Path to a bundled jar including your application and all dependencies.

    `application-arguments`: Arguments passed to the main method of your main class, if any.

### Usage Examples
#### Submit Spark-Pi job (spark native mode)

<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/174703141-63209559-05e1-4c4d-b096-6b862a9bed8a.png" alt="data lifecycle" width='250px' />
</p>

```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --sgx-enabled false \
        --master local[2] \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```
#### Submit Spark-Pi job (spark native mode, sgx enabled)
<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/174703165-2afc280d-6a3d-431d-9856-dd5b3659214a.png" alt="data lifecycle" width='250px' />
</p>

```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --master local[2] \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
#### Submit Spark-Pi job (k8s client mode, sgx enabled)
<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/174703216-70588315-7479-4b6c-9133-095104efc07d.png" alt="data lifecycle" width='500px' />
</p>

```
#!/bin/bash

export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```

If you want to enable the spark security configurations as in [Spark security configurations](https://github.com/intel-analytics/BigDL/edit/main/ppml/trusted-big-data-ml/python/docker-graphene/README.md#2-spark-security-configurations), export secure_password before invokeing PPML CLI to enable it.
```
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
```

#### Submit Spark-Pi job (k8s cluster mode, sgx enabled)
<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/174703234-e45b8fe5-9c61-4d17-93ef-6b0c961a2f95.png" alt="data lifecycle" width='500px' />
</p>

```
#!/bin/bash

export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
If you want to enable the spark security configurations as in [Spark security configurations](https://github.com/intel-analytics/BigDL/edit/main/ppml/trusted-big-data-ml/python/docker-graphene/README.md#2-spark-security-configurations), export secure_password before invokeing PPML CLI to enable it.
```
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
```

## Helm Chart

https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/kubernetes/README.md
