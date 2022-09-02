# Trusted Big Data ML with Occlum

## Resource Configuration Guide
These configuration values must be tuned on a per-application basis.
You can refer to [here](https://github.com/occlum/occlum/blob/master/docs/resource_config_guide.md?plain=1) for more information.
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=24GB  // means the whole image memory you can use, the same as resource_limits.user_space_size
-e SGX_THREAD=512  // means the whole thread you can use, the same as resource_limits.max_num_of_threads
-e SGX_HEAP=512MB  // means each process init malloc memory, the same as process.default_heap_size
-e SGX_KERNEL_HEAP=1GB // means occlum in kernel state using memory, the same as resource_limits.kernel_space_heap_size
```
the log of Occlum can be turned on by setting the `OCCLUM_LOG_LEVEL` environment variable (e.g.,
`OCCLUM_LOG_LEVEL=error`, `OCCLUM_LOG_LEVEL=info`, `OCCLUM_LOG_LEVEL=trace`).
You can add 'OCCLUM_LOG_LEVEL=trace' in [run_spark_on_occlum_glibc.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/run_spark_on_occlum_glibc.sh#L3) and change set this [config](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/run_spark_on_occlum_glibc.sh#L46) true without " ".

## Prerequisites

Pull image from dockerhub.

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.1.0-SNAPSHOT
```

Also, you can build image with `build-docker-image.sh`. Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:

``` bash
bash build-docker-image.sh
```

## Before run example

### Check Device
Before run any example, please make sure you have correctly set the --device option in the start-spark-local.sh according to your machine.

For example:
```
  --device=/dev/sgx
```
or:
```
  --device=/dev/sgx/enclave
  --device=/dev/sgx/provision
```

## Spark 3.1.2 Pi example

To run Spark Pi example, start the docker container with:

``` bash
bash start-spark-local.sh pi
```

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=6GB \
-e SGX_THREAD=512 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can see Pi result in logs (`docker logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`)

```bash
Pi is roughly 3.1436957184785923
```

## BigDL Lenet Mnist Example

To train a model with PPML in BigDL, you need to prepare the data first. You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). There are 5 files in total. `train-images-idx3-ubyte` contains train images; `train-labels-idx1-ubyte` is the train label file; `t10k-images-idx3-ubyte` has validation images; `t10k-labels-idx1-ubyte` contains validation labels. Unzip all the files and put them in a new directory `data`.

**By default, `data` dir will be mounted to `/opt/occlum_spark/data` in container (become `/host/data` in occlum). You can change data path in `start-spark-local.sh`.**

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=60GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

To run BigDL Lenet Mnist example, start the docker container with:

``` bash
bash start-spark-local.sh lenet -b 4 -e 1
```

The examples are run in the docker container. Attach it and see the results (`docker attach logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`). `-b 4 -e 1` means batch size 4 and epoch 1.

```bash
2021-10-29 01:57:48 INFO  DistriOptimizer$:431 - [Epoch 1 40/60000][Iteration 10][Wall Clock 14.768519551s] Trained 4.0 records in 0.348563287 seconds. Throughput is 11.475678 records/second. Loss is 2.4064577. Sequentialc3a85127s hyper parameters: Current learning rate is 0.05. Current dampening is 1.7976931348623157E308.
2021-10-29 01:57:48 WARN  DistriOptimizer$:238 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
```

## BigDL Resnet Cifar-10 Example

Download the Cifar-10 dataset (CIFAR-10 binary version) from [here](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset contains 5 files, i.e, `data_batch_1.bin`, `data_batch_2.bin`, `data_batch_3.bin`, `data_batch_4.bin`, `data_batch_5.bin` and `test_batch.bin`. Put all the files in `data` directory.

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=30GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=4GB \
```

To run BigDL ResNet CIFAR-10 example, start the docker container with:

``` bash
bash start-spark-local.sh resnet --batchSize 400 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 1 --learningRate 0.1 
```

The examples are run in the docker container. Attach it and see the results (`docker attach logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`). `--batchSize 400 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 1 --learningRate 0.1` are resent training related parameters.


```bash
2021-10-29 02:32:24 INFO  DistriOptimizer$:431 - [Epoch 5 400/50000][Iteration 501][Wall Clock 957.374866852s] Trained 400.0 records in 1.772934817 seconds. Throughput is 225.61461 records/second. Loss is 0.72196096. Sequentialf702431bs hyper parameters: Current learning rate is 0.1. Current weight decay is 1.0E-4. Current momentum is 0.9. Current nesterov is true.
2021-10-29 02:32:25 INFO  DistriOptimizer$:431 - [Epoch 5 800/50000][Iteration 502][Wall Clock 959.109945609s] Trained 400.0 records in 1.735078757 seconds. Throughput is 230.53708 records/second. Loss is 0.8222007. Sequentialf702431bs hyper parameters: Current learning rate is 0.1. Current weight decay is 1.0E-4. Current momentum is 0.9. Current nesterov is true.
2021-10-29 02:32:27 INFO  DistriOptimizer$:431 - [Epoch 5 1200/50000][Iteration 503][Wall Clock 960.971331791s] Trained 400.0 records in 1.861386182 seconds. Throughput is 214.89362 records/second. Loss is 0.5886179. Sequentialf702431bs hyper parameters: Current learning rate is 0.1. Current weight decay is 1.0E-4. Current momentum is 0.9. Current nesterov is true.
```

## Spark TPC-H example


You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=24GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

### Generate Data

```
git clone https://github.com/intel-analytics/zoo-tutorials.git && \
cd zoo-tutorials/tpch-spark/dbgen && \
make
```

Then you can generate 1G size data by:
```
./dbgen -s 1
```

Then mount `/path/to/zoo-tutorials/tpch-spark/dbgen` to container's `/opt/occlum_spark/data` in `start-spark-local.sh` via:
```
-v /path/to/zoo-tutorials/tpch-spark/dbgen:/opt/occlum_spark/data
```

Start run spark tpc-h example:
```
bash start-spark-local.sh tpch
```

You will find `output` folder under `/path/to/zoo-tutorials/tpch-spark/dbgen` which contains sql result.

## Spark SQL Scala Unit Tests

### Run Spark SQl Scala Unit Tests

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=60GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

To run Spark Sql Scala Unit Tests, start the docker container with:
```
bash start-spark-local.sh ut
```
You can see some output like this:
```
22/01/28 03:06:54 INFO SqlResourceSuite: 

===== TEST OUTPUT FOR o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = false and planDescription = false' =====

22/01/28 03:06:54 INFO SqlResourceSuite: 

===== FINISHED o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = false and planDescription = false' =====

22/01/28 03:06:54 INFO SqlResourceSuite: 

===== TEST OUTPUT FOR o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = true and planDescription = false' =====

22/01/28 03:06:54 INFO SqlResourceSuite: 

===== FINISHED o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = true and planDescription = false' =====

22/01/28 03:06:54 INFO SqlResourceSuite: 

===== TEST OUTPUT FOR o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = true and planDescription = true' =====

22/01/28 03:06:54 INFO SqlResourceSuite: 

===== FINISHED o.a.s.status.api.v1.sql.SqlResourceSuite: 'Prepare ExecutionData when details = true and planDescription = true' =====
```
And the log files will be saved to `data/olog` folder.

## BigDL XGBoost Example

### Download data
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Split 10g data from the dataset and put it into a folder. Then mount `/path/to/data/10g_data` to container's `/opt/occlum_spark/data` in `start-spark-local.sh` via:
```
-v /path/to/data/10g_data:/opt/occlum_spark/data
```

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=30GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-i /host/data        // -i means inputpath of training data
-s /host/data/model  // -s means savepath of model
-t 2                 // -t means threads num
-r 100               // -r means Round num
-d 2                 // -d means maxdepth
-w 1                 // -w means Workers num
```

Start run BigDL Spark XGBoost example:
```
bash start-spark-local.sh xgboost
```

The console output looks like:
```
[INFO] [02/10/2022 14:57:04.244] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [0]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:07.296] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [1]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:10.071] [RabitTracker-akka.actor.default-dispatcher-7] [akka://RabitTracker/user/Handler] [2]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
```

You can find XGBoost model under folder `/path/to/data/`.
```
/path/to/data/
├── data
│   └── XGBoostClassificationModel
└── metadata
    ├── part-00000
    └── _SUCCESS
```

## How to debug
Modify the `SGX_LOG_LEVEL` to one of `off, error, warn, debug, info, and trace` in `start-spark-local.sh`. 
The default value is off, showing no log messages at all. The most verbose level is trace.
When you use attestation, `SGX_LOG_LEVEL` will be set to `off`.

## Start BigDL PPML Occlum Attestation Server
Modify `PCCL_URL`, `ATTESTATION_SERVER_IP` and `ATTESTATION_SERVER_PORT` in `start-occlum-attestation-server.sh`, Then
```commandline
bash start-occlum-attestation-server.sh
```
You will see:
```
Server listening on $ATTESTATION_SERVER_IP:$ATTESTATION_SERVER_PORT
```

Get `image_key`:
```commandline
docker cp bigdl-ppml-trusted-big-data-ml-scala-occlum-attestation-server:/root/demos/remote_attestation/init_ra_flow/image_key ./data
```

## Before you run examples, you need to mount this `image_key` to container's `/opt/occlum_spark/data/`. We have already done it for you.
