# Spark 3.1.2 on K8S with Occlum

## Resource Configuration Guide
some configuration in *.yaml is introduced in [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/README.md), you can refer to it for more information.
The two new configs 'spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE' and 'spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE' are the same as driver-memory and executor-memory in spark. We use original driver-memory and original executor-memory to alloc extra common memory for libos.   You can refer to [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#configuration-explainations) for more information.

### Linux glibc2.12+ Arena thread memory pool
The default value of MALLOC_ARENA_MAX in linux is the number of CPU cores * 8, and it will cost most MALLOC_ARENA_MAX * 128M EPC in SGX.
So we set MALLOC_ARENA_MAX=1 by default to disable it to reduce EPC usage, but sometimes it may makes the performance a little bit worse.
After our many tests, Using MALLOC_ARENA_MAX=1 is fine in most cases. Hadoop recommend to set MALLOC_ARENA_MAX=4, you can set it by this way:
```bash
--conf spark.kubernetes.driverEnv.MALLOC_ARENA_MAX=4 \
--conf spark.executorEnv.MALLOC_ARENA_MAX=4 \
```

## Prerequisite

* Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)
* Prepare image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.3.0-SNAPSHOT`

1. Pull image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.3.0-SNAPSHOT
```

If Dockerhub is not accessible, we can build a docker image with Dockerfile and modify the path in the build-docker-image.sh firstly.

``` bash
cd ..
bash build-docker-image.sh
```

2. Download [Spark 3.1.2](https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz), and setup `SPARK_HOME`.
3. `export kubernetes_master_url=your_k8s_master` or replace `${kubernetes_master_url}` with your k8s master URL in `run_spark_xxx.sh`.
4. Modify `driver.yaml` and `executor.yaml` for your applications.
   In our demo example, we mount SGX devices into a container or pod. Mount device requires privileged: true. In a production deployment, please use K8S SGX device plugin with the device-plugin setting in yaml.

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
We recommend using hdfs to read input-data and write output-result instead of mouting data.

## BigDL LGBM Example

### Download data
You can download the iris.data from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
Put it into folder `/tmp/iris.data`. You can change the path to data via change mount path `data-exchange` in `executor.yaml` and `driver.yaml`.
We recommend using hdfs to read input-data and write output-result instead of mouting data.
And you can find the source code [here](https://github.com/intel-analytics/BigDL/blob/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/lightGBM/LgbmClassifierTrain.scala) and use more configs.for example:
```
--inputPath hdfs://IP/input/iris/iris.data \
--numIterations 100 \
--partition 8 \
--modelSavePath hdfs://IP/input/output/iris
```

### Generate SSL keys and ertificate
You can get bash from [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/scripts/generate-ssl.sh).
Then generate your ssl key and certificate in /ppml/keys, and mount it to `/opt/occlum_spark/image/ppml` in pods.
If you run this examples in local node, you can mount like the `data-exchange`.
Or you can use nfs server to mount:
```
#driver.yaml  and executor.yaml
volumeMounts:
- name: nfs-data
  mountPath: /opt/occlum_spark/image/ppml
volumes:
  - name: nfs-data
    nfs:
      server: your_IP
      path: /ppml
```

Then:
```bash
./run_spark_lgbm.sh
```
Note that if you do not have ssl key and certificate in `/ppml/keys`, the distributed lgbm training will failed like this:
```
[LightGBM] [Warning] Unable to set TLSV2 cipher, ignored and try to set TLSV3...
error:1410D0B9:SSL routines:SSL_CTX_set_cipher_list:no cipher match
error:1426E0B9:SSL routines:ciphersuite_cb:no cipher match
```

#### Source code
You can find the source code [here](https://github.com/intel-analytics/BigDL/tree/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/gbt/gbtClassifierTrainingExampleOnCriteoClickLogsDataset).

### Run Spark TPC-H example

Generate 1g Data like [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#generate-data), and you can use hdfs to replace the mount way, and you can just execute one query by adding [query_number] from 1 to 22 behind output_dir.For example:
"hdfs:///input/dbgen hdfs:///output/dbgen 13" means execute query 13.

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

## PySpark Example
The following three simple PySpark examples can run directly.
If you want to run another example, please make sure that PPML-Occlum image have installed the relevant dependencies.
```bash
#default dependencies
-y python=3.8.10 numpy=1.21.5 scipy=1.7.3 scikit-learn=1.0 pandas=1.3 Cython
```
And upload source file by hdfs in the last line to replace local file.For example:
```bash
#run_pyspark_sql_example.sh
    #local:/py-examples/sql_example.py
    hdfs://${IP}:${PORT}/${PATH}/sql_example.py
```
### PySparkPi example

```bash
./run_pyspark_pi.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "512m"
    - name: SGX_MEM_SIZE
      value: "10GB"
    - name: SGX_THREAD
      value: "512"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "10GB"
    - name: SGX_THREAD
      value: "512"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

### PySpark SQL example

```bash
./run_pyspark_sql_example.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "1g"
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "1024"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "1024"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

### PySpark sklearn LinearRegression example

```bash
./run_pyspark_sklearn_example.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "1g"
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "1024"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "1024"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

### PySpark TPC-H example
Generate 1g Data like [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum#generate-data), and you can use hdfs to replace the mount way, and you can just execute one query by adding [query_number] from 1 to 22 behind output_dir.For example:
"hdfs:///input/dbgen/ hdfs:///output/dbgen/ true 13" means using SQL directly execute query 13.
```bash
./run_pyspark_tpch.sh
```

```yaml
#driver.yaml
    env:
    - name: DRIVER_MEMORY
      value: "512m"
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "512"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

```yaml
#executor.yaml
    env:
    - name: SGX_MEM_SIZE
      value: "15GB"
    - name: SGX_THREAD
      value: "512"
    - name: SGX_HEAP
      value: "1GB"
    - name: SGX_KERNEL_HEAP
      value: "1GB"
```

### [Deprecated] Spark XGBoost example

> Warning: Running XGBoost in distributed mode is not safe due to the fact that Rabit's network (which contains gradient, split, and env) is not protected.

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

## Using BigDL PPML Occlum EHSM Attestation Server in k8s
Bigdl ppml use EHSM as reference KMS&AS, you can deploy EHSM following the [guide](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#deploy-bigdl-ehsm-kms-on-kubernetes-with-helm-charts)
We assume you have already set up environment and enroll yourself on EHSM.

In driver.yaml and executor.yaml. Set `ATTESTATION` = true and modify `PCCL_URL`, `ATTESTATION_URL` to the env value you have set,
and modify `APP_ID`, `API_KEY` to the value you have get  when enroll, and then you can change `CHALLENGE` and
`REPORT_DATA` for attestation. But now in k8s it's not perfect, causing each executor need to register again unnecessarily.

``` yaml
#*.yaml
- name: ATTESTATION   #set to true to start attestation.
  value: false   
- name: PCCS_URL   #PCCS URL, obtained from KMS services or a self-deployed one. Should match the format https://<ip_address>:<port>.
  value: "https://PCCS_IP:PCCS_PORT"  
- name: ATTESTATION_URL   #URL of attestation service. Should match the format <ip_address>:<port>.
  value: "ESHM_IP:EHSM_PORT"
- name: APP_ID   #The appId generated by your attestation service.
  value: your_app_id
- name: API_KEY   #The apiKey generated by your attestation service.
  value: your_api_key
- name: CHALLENGE   #Challenge to get quote of attestation service which will be verified by local SGX SDK. Should be a BASE64 string. It can be a casual BASE64 string, for example, it can be generated by the command echo ppml|base64.
  value: cHBtbAo=
- name: REPORT_DATA   #A random String to generator a quote which will be send to attestation service and use for attest.
  value: ppml
```