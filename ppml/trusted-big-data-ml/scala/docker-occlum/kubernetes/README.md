# Spark 3.1.2 on K8S with Occlum

## Resource Configuration Guide
some configuration in *.yaml is introduced in [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/README.md), you can refer to it for more information.
The two new configs 'spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE' and 'spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE' are the same as driver-memory and executor-memory in spark. We use original driver-memory and original executor-memory to alloc extra common memory for libos.   You can refer to [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#configuration-explainations) for more information.
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

2. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`.
3. `export kubernetes_master_url=your_k8s_master` or replace `${kubernetes_master_url}` with your k8s master url in `run_spark_xxx.sh`.
4. Modify `driver.yaml` and `executor.yaml` for your applications.
   In our demo example, we mount SGX devices into container or pod. Mount device requires privileged: true. In production deployment, please use K8S SGX device plugin with device-plugin setting in yaml.

## Examples

### SparkPi example

```bash
./run_spark_pi.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "512m"
    - name: SGX_MEM_SIZE
      value: "8GB"
    - name: SGX_THREAD
      value: "256"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "8GB"
    - name: SGX_THREAD
      value: "256"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

### Spark ML LogisticRegression example

```bash
./run_spark_lr.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "2g"
    - name: SGX_MEM_SIZE
      value: "20GB"
    - name: SGX_THREAD
      value: "512"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "10GB"
    - name: SGX_THREAD
      value: "512"
```

### Spark ML GradientBoostedTreeClassifier example

```bash
./run_spark_gbt.sh
```

### Spark SQL SparkSQL example

```bash
./run_spark_sql.sh
```

### Run Spark GBTClassifier example using CriteoClickLogsDataset

#### Criteo 1TB Click Logs [dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)

Split 1g dataset and put it into folder `/tmp/gbt_data`. 
You can change the path to data via change mount path `data-exchange` in `executor.yaml`.
Then:
```bash
./run_spark_gbt_criteo.sh
```

Parameters:

* -i means input_path : String.

    For example, yout host path to Criteo dateset is `/tmp/gbt_data/criteo` then this parameter in `run_spark_gbt_criteo.sh` is `/host/data/gbt_data`.
* -s means save_path : String.

    After training, you can find gbt result in folder `/tmp/path_to_save`.

* -I means max_Iter : Int
* -d means max_depth: Int.
We recommend to use hdfs to read input-data and write output-result instead of mouting data.

#### Source code
You can find source code [here](https://github.com/intel-analytics/BigDL/tree/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/gbt/gbtClassifierTrainingExampleOnCriteoClickLogsDataset).

### Run Spark TPC-H example

Generate 1g Data like [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#generate-data), and you can use hdfs to replace the mount way, and you can just excute one query by adding [query_number] from 1 to 22 behind output_dir.For example:
"hdfs:///input/dbgen hdfs:///output/dbgen 13" means excute query 13.

Modify the following configuration in 'driver.yaml' and 'executor.yaml' and 'run_spark_tpch.sh'.

```yaml
#driver.yaml
env:
- name: DRIVER_MEMORY
  value: "1g"
- name: SGX_MEMORY_SIZE
  value: "10GB"
- name: SGX_THREAD
  value: "1024"
- name: SGX_HEAP
  value: "1GB"
- name: SGX_KERNEL_HEAP
  value: "2GB"
- name: META_SPACE
  value: "1024m"
```

```yaml
#excutor.yaml
env:
- name: SGX_MEMORY_SIZE
  value: "10GB"
- name: SGX_THREAD
  value: "1024"
- name: SGX_HEAP
  value: "1GB"
- name: SGX_KERNEL_HEAP
  value: "2GB"
```

```bash
#run_spark_tpch.sh
--num-executors 2 \
--executor-cores 4 \
--executor-memory 4g \
```

Or you can directly add the following configuration in [run_spark_tpch.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/kubernetes/run_spark_tpch.sh) and it will overwrite the changes in *.yaml.

```bash
#run_spark_tpch.sh
    --conf spark.kubernetes.driverEnv.DRIVER_MEMORY=1g \
    --conf spark.kubernetes.driverEnv.SGX_MEM_SIZE="10GB" \
    --conf spark.kubernetes.driverEnv.META_SPACE=1024m \
    --conf spark.kubernetes.driverEnv.SGX_HEAP="1GB" \
    --conf spark.kubernetes.driverEnv.SGX_KERNEL_HEAP="2GB" \
    --conf spark.kubernetes.driverEnv.SGX_THREAD="1024" \
    --conf spark.executorEnv.SGX_MEM_SIZE="10GB" \
    --conf spark.executorEnv.SGX_KERNEL_HEAP="1GB" \
    --conf spark.executorEnv.SGX_HEAP="1GB" \
    --conf spark.executorEnv.SGX_THREAD="1024" \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 4g \
```


Then run the script.

```bash
./run_spark_tpch.sh
```

### [Deprecated] Spark XGBoost example
Distributed XGBoost it doesn't support....due to lack of network proteciton.

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

Split 1G data from this dataset and put it into `/tmp/xgboost_data`. 
Then change the `class` in [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/kubernetes/run_spark_xgboost.sh#L7) to
`com.intel.analytics.bigdl.dllib.example.nnframes.xgboost.xgbClassifierTrainingExampleOnCriteoClickLogsDataset`.

Add these configurations to [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/kubernetes/run_spark_xgboost.sh):

```bash
    --conf spark.driver.extraClassPath=local:///opt/spark/jars/* \
    --conf spark.executor.extraClassPath=local:///opt/spark/jars/* \
    --conf spark.task.cpus=6 \
    --conf spark.cores.max=12 \
    --conf spark.executor.instances=2 \
    --conf spark.kubernetes.driverEnv.DRIVER_MEMORY=1g \
    --conf spark.kubernetes.driverEnv.SGX_MEM_SIZE="12GB" \
    --conf spark.kubernetes.driverEnv.META_SPACE=1024m \
    --conf spark.kubernetes.driverEnv.SGX_HEAP="1GB" \
    --conf spark.kubernetes.driverEnv.SGX_KERNEL_HEAP="2GB" \
    --conf spark.executorEnv.SGX_MEM_SIZE="10GB" \
    --conf spark.executorEnv.SGX_KERNEL_HEAP="1GB" \
    --conf spark.executorEnv.SGX_HEAP="1GB" \
    --executor-cores 6 \
    --executor-memory 3g \
    --driver-memory 1g \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE_NO="3G" \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="1G" 
```

Change the `parameters` to:

```commandline
-i /host/data/xgboost_data -s /host/data/xgboost_criteo_model -t 32 -r 100 -d 10 -w 2
```

Then:

```bash
./run_spark_xgboost.sh
```
Parameters:

* -i means inputpath_to_Criteo_data : String.

    For example, yout host path to Criteo dateset is `/tmp/xgboost_data/criteo` then this parameter in `run_spark_xgboost.sh` is `/host/data/xgboost_data`.
* -s means savepath_to_model_to_be_saved : String.

    After training, you can find xgboost model in folder `/tmp/path_to_model_to_be_saved`.

* -t means num_threads : Int
* -r means num_round : Int
* -d means max_depth: Int.
* -w means num_workers: Int.

**Note: make sure num_threads is no larger than spark.task.cpus.**

## How to debug

Modify the `--conf spark.kubernetes.sgx.log.level=off \` to one of `debug or trace` in `run_spark_xx.sh`.
