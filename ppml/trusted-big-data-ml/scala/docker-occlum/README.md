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

### Linux glibc2.12+ Arena thread memory pool
The default value of MALLOC_ARENA_MAX in linux is the number of CPU cores * 8, and it will cost most MALLOC_ARENA_MAX * 128M EPC in SGX.
So we set MALLOC_ARENA_MAX=1 by default to disable it to reduce EPC usage, but sometimes it may makes the performance a little bit worse.
After our many tests, Using MALLOC_ARENA_MAX=1 is fine in most cases. Hadoop recommend to set MALLOC_ARENA_MAX=4, you can set it by this way:
```bash
-e MALLOC_ARENA_MAX=4
```

## Prerequisites

Pull image from dockerhub.

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.3.0-SNAPSHOT
```

Also, you can build the image with `build-docker-image.sh`. Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:

``` bash
bash build-docker-image.sh
```

## Before running the example

### Check Device
Before running any example, please make sure you have correctly set the --device option in the start-spark-local.sh according to your machine.

For example:
```
  --device=/dev/sgx
```
or:
```
  --device=/dev/sgx/enclave
  --device=/dev/sgx/provision
```

## Spark 3.1.3 Pi example

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

**By default, `data` dir will be mounted to `/opt/occlum_spark/data` in container (become `/host/data` in occlum). You can change the data path in `start-spark-local.sh`.**

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
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Split 1g of data from the dataset and put it into a folder. Then mount `/path/to/data/1g_data` to container's `/opt/occlum_spark/data` in `start-spark-local.sh` via:
```
-v /path/to/data/1g_data:/opt/occlum_spark/data
```

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=30GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can change the configuration If you enter image and run /opt/run_spark_on_occlum_glibc.sh xgboost in docker container.
``` bash
#run_spark_on_occlum_glibc.sh
#run_spark_xgboost()
-i /host/data        // -i means inputpath of training data
-s /host/data/model  // -s means savepath of model
-t 2                 // -t means threads num
-r 100               // -r means Round num
-d 2                 // -d means maxDepth
-w 1                 // -w means Workers num
```

Start run BigDL Spark XGBoost example:
```
bash start-spark-local.sh xgboost
```

The console output looks like this:
```
[INFO] [02/10/2022 14:57:04.244] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [0]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:07.296] [RabitTracker-akka.actor.default-dispatcher-3] [akka://RabitTracker/user/Handler] [1]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
[INFO] [02/10/2022 14:57:10.071] [RabitTracker-akka.actor.default-dispatcher-7] [akka://RabitTracker/user/Handler] [2]  train-merror:0.030477   eval1-merror:0.030473   eval2-merror:0.030350
```

You can find XGBoost model under the folder `/path/to/data/`.
```
/path/to/data/
├── data
│   └── XGBoostClassificationModel
└── metadata
    ├── part-00000
    └── _SUCCESS
```

## BigDL GBT Example

### Download data
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Split 1g of data from the dataset and put it into a folder. Then mount `/path/to/data/1g_data` to container's `/opt/occlum_spark/data` in `start-spark-local.sh` via:
```
-v /path/to/data/1g_data:/opt/occlum_spark/data
```

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=30GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can change the configuration If you enter image and run /opt/run_spark_on_occlum_glibc.sh gbt in docker container.
``` bash
#start-spark-local.sh
#run_spark_gbt()
-i /host/data        // -i means inputpath of training data
-s /host/data/model  // -s means savepath of model
-I 100               // -r means maxInter
-d 5                 // -d means maxDepth
```

Start run BigDL Spark GBT example:
```
bash start-spark-local.sh gbt
```

You can find GBT result under folder `/path/to/data/model/`.
```
/path/to/data/model/
├── data
├── treesMetadata
└── metadata
    ├── part-00000
    └── _SUCCESS
```

## BigDL LGBM Example

### Download data
You can download the iris.data from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).Then mount `/path/to/data/iris.data` to container's `/opt/occlum_spark/data` in `start-spark-local.sh` via:
```
-v /path/to/data/:/opt/occlum_spark/data
```

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=15GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can change the configuration If you enter image and run /opt/run_spark_on_occlum_glibc.sh lgbm in docker container. The source code is [here](https://github.com/intel-analytics/BigDL/blob/main/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/lightGBM/LgbmClassifierTrain.scala).
You can see it and use more configuration.
``` bash
#start-spark-local.sh
#run_spark_lgbm()
--inputPath /host/data/iris.data \
--numIterations 100 \
--partition 4 \
--modelSavePath /host/data/iris_output
```

Start run BigDL Spark LGBM example:
```
bash start-spark-local.sh lgbm
```

You can find lgbm result under folder `/path/to/data/iris_model/`.
```
/path/to/data/iris_model/*.txt
```

## BigDL GBT e2e Example

### Download data
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Split 1g of data from the dataset and put it into a folder. Then mount `/path/to/data/` to container's `/opt/occlum_spark/data/` in `start-spark-local.sh` via:
```
-v /path/to/data/gbt/1g_data:/opt/occlum_spark/data/gbt/1g_dara
```

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
-e PCCS_URL=https://PCCS_IP:PCCS_PORT \
-e ATTESTATION_URL=ESHM_IP:EHSM_PORT \
-e APP_ID=your_app_id \
-e API_KEY=your_api_key \
```

You can change the configuration If you enter image and run /opt/run_spark_on_occlum_glibc.sh gbt_e2e in docker container.
``` bash
#start-spark-local.sh
-i /host/data/encrypt  // -i means inputpath of encrypt training data
-s /host/data/model  // -s means savepath of model
-I 100               // -r means maxInter
-d 5                 // -d means maxDepth
```

Start run BigDL Spark GBT e2e example:

1.Input PCCS_URL,ATTESTATION_URL,APP_ID and API_KEY first. Change the file [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh) last line from `bash /opt/run_spark_on_occlum_glibc.sh $1` to `bash`
And then run `bash start-spark-local.sh` to enter docker container.
```
bash start-spark-local.sh
```
2.To generate primary key for encrypt and decrypt. The primary key will be generated in `/opt/occlum_spark/data/key/ehsm_encrypted_primary_key`.
```
bash /opt/ehsm_entry.sh generatekey ehsm $APP_ID $API_KEY
```
3.To encrypt input data. For example, you mount a file called day_0_1g.csv. It will be encrypted in `/opt/occlum_spark/data/encryptEhsm`.
```
bash /opt/ehsm_entry.sh  encrypt ehsm $APP_ID $API_KEY /opt/occlum_spark/data/gbt/day_0_1g.csv
```
4.To run the BigDL GBT e2e Example.
```
bash /opt/run_spark_on_occlum_glibc.sh gbt_e2e
```
You can find GBT result under folder `/opt/occlum_spark/data/model/`.
```
/opt/occlum_spark/data/model/
├── data
├── treesMetadata
└── metadata
    ├── part-00000
    └── _SUCCESS
```

## BigDL SimpleQuery e2e Example

You can enlarge the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
-e PCCS_URL=https://PCCS_IP:PCCS_PORT \
-e ATTESTATION_URL=ESHM_IP:EHSM_PORT \
-e APP_ID=your_app_id \
-e API_KEY=your_api_key \
```

Start run BigDL Spark SimpleQuery e2e example:

1.Input PCCS_URL,ATTESTATION_URL,APP_ID and API_KEY first. Change the file [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh) last line from `bash /opt/run_spark_on_occlum_glibc.sh $1` to `bash`
And then run `bash start-spark-local.sh` to enter docker container.
```
bash start-spark-local.sh
```
2.To generate primary key for encrypt and decrypt. The primary key will be generated in `/opt/occlum_spark/data/key/ehsm_encrypted_primary_key`.
```
bash /opt/ehsm_entry.sh generatekey ehsm $APP_ID $API_KEY
```
3.To generate input data
you can use [generate_people_csv.py](https://github.com/intel-analytics/BigDL/tree/main/ppml/scripts/generate_people_csv.py). The usage command of the script is:
```bash
python generate_people_csv.py /opt/occlum_spark/data/people.csv <num_lines>
```
4.To encrypt input data. For example, you mount a file called people.csv. It will be encrypted in `/opt/occlum_spark/data/encryptEhsm`.
```
bash /opt/ehsm_entry.sh  encrypt ehsm $APP_ID $API_KEY /opt/occlum_spark/data/people.csv
```
5.To run the BigDL SimpleQuery e2e Example.
```
bash /opt/run_spark_on_occlum_glibc.sh sql_e2e
```
6.You can find encrypted result under folder `/opt/occlum_spark/data/model`. And decrypt the result by:
```
bash /opt/ehsm_entry.sh  decrypt ehsm $APP_ID $API_KEY /opt/occlum_spark/data/model
```
And the decrypt result is under folder `/opt/occlum_spark/data/decryptEhsm`.

## BigDL MultiPartySparkQuery e2e Example

You can set the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
-e PCCS_URL=https://PCCS_IP:PCCS_PORT \
-e ATTESTATION_URL=ESHM_IP:EHSM_PORT \
-e APP_ID=your_app_id \
-e API_KEY=your_api_key \
```

Start run BigDL MultiParty Spark Query e2e example:

1.Input PCCS_URL,ATTESTATION_URL,APP_ID and API_KEY first. Change the file [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh) last line from `bash /opt/run_spark_on_occlum_glibc.sh $1` to `bash`
And then run `bash start-spark-local.sh` to enter docker container.
```
bash start-spark-local.sh
```
2.To generate primary key for encrypt and decrypt. We have set the value of APP_ID and API_KEY = `123456654321` for simple KMS.
The EHSM primary key will be generated in `/opt/occlum_spark/data/key/ehsm_encrypted_primary_key`. The simple KMS primary key will be generated in `/opt/occlum_spark/data/key/simple_encrypted_primary_key`
```
bash /opt/ehsm_entry.sh generatekey ehsm $APP_ID $API_KEY
bash /opt/ehsm_entry.sh generatekey simple $APP_ID $API_KEY
```
3.To generate input data
you can use [generate_people_csv.py](https://github.com/intel-analytics/BigDL/tree/main/ppml/scripts/generate_people_csv.py). The usage command of the script is:
```bash
python generate_people_csv.py /opt/occlum_spark/data/Amy.csv <num_lines>
python generate_people_csv.py /opt/occlum_spark/data/Bob.csv <num_lines>
```
4.To encrypt input data.Using EHSM, the Bob.csv will be encrypted in `/opt/occlum_spark/data/encryptEhsm`. Using simple KMS, the Bob.csv will be encrypted in `/opt/occlum_spark/data/encryptSimple`. For example:
```
bash /opt/ehsm_entry.sh  encrypt ehsm $APP_ID $API_KEY /opt/occlum_spark/data/Bob.csv
bash /opt/ehsm_entry.sh  encrypt simple $APP_ID $API_KEY /opt/occlum_spark/data/Amy.csv
```
5.To run the BigDL MultiParty Spark Query e2e Example.
```
bash /opt/run_spark_on_occlum_glibc.sh multi_sql_e2e
```
6.You can find encrypted result under folder `/opt/occlum_spark/data/unoin_output` and `/opt/occlum_spark/data/join_output`.
 And decrypt the result by:
```
bash /opt/ehsm_entry.sh  decrypt simple $APP_ID $API_KEY /opt/occlum_spark/data/union_output
bash /opt/ehsm_entry.sh  decrypt ehsm $APP_ID $API_KEY /opt/occlum_spark/data/join_output
```
And the decrypt result is under folder `/opt/occlum_spark/data/decryptSimple` and `/opt/occlum_spark/data/decryptEhsm`.

## PySpark 3.1.3 Pi example

To run PySpark Pi example, start the docker container with:

``` bash
bash start-spark-local.sh pypi
```

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=10GB \
-e SGX_THREAD=512 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can see Pi result in logs (`docker logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`)

```bash
Pi is roughly 3.1436957184785923
```

## PySpark 3.1.3 SQL example

To run PySpark SQL example, start the docker container with:

``` bash
bash start-spark-local.sh pysql
```

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

## PySpark sklearn LinearRegression example

To run PySpark sklearn example, start the docker container with:

``` bash
bash start-spark-local.sh pysklearn
```

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=1024 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

You can see result in logs (`docker logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`)

## PySpark TPC-H example

You can change the configuration in [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh)
``` bash
#start-spark-local.sh
-e SGX_MEM_SIZE=15GB \
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
bash start-spark-local.sh pytpch
```

You will find `output` folder under `/path/to/zoo-tutorials/tpch-spark/dbgen` which contains sql result.

## How to debug
Modify the `SGX_LOG_LEVEL` to one of `off, debug and trace` in `start-spark-local.sh`. 
The default value is off, showing no log messages at all. The most verbose level is trace.
When you use attestation, `SGX_LOG_LEVEL` will be set to `off`.

## How to enabled hdfs encryption service
You can refer to [here](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/TransparentEncryption.html) for more information.
### How to start hadoop KMS service
1.	Make sure you can correctly start and use hdfs
2.	To config a KMS client in $HADOOP_HOME/etc/hadoop/core-site.xml(If your hdfs is running in a distributed system, you need to update all nodes.), for example:
```xml
<property>
    <name>hadoop.security.key.provider.path</name>
    <value>kms://http@172.168.0.205:9600/kms</value>
    <description>
        The KeyProvider to use when interacting with encryption keys used
        when reading and writing to an encryption zone.
    </description>
</property>
```
3. To config the KMS backing KeyProvider properties in the $HADOOP_HOME/etc/hadoop/kms-site.xml configuration file. 
```xml
<property>
    <name>hadoop.kms.key.provider.uri</name>
    <value>jceks://file@/${user.home}/kms.keystore</value>
</property>
```
4. Restart you hdfs server. 
```bash
sbin/stop-dfs.sh  sbin/start-dfs.sh
```
5. Start KMS server. 
```bash
hadoop --daemon start|stop kms
```
6. Run this bash command to check if the KMS started 
```bash
hadoop key list
```

### How to use KMS to encrypt and decrypt data
1. Create a new encryption key for an encryption zone
```bash
hadoop key create mykey
```
2.	Create a new empty directory(must) and make it an encryption zone
```bash
hdfs crypto -createZone -keyName mykey -path /empty_zone
```
3.	Get encryption information from the file
```bash
hdfs crypto -getFileEncryptionInfo -path /empty_zone/helloWorld
```
4. Add permission control to users or groups in $HADOOP_HOME/etc/hadoop/kms-acls.xml. It will be hotbooted after every update. For example:
```xml
<property>
    <name>key.acl.mykey.ALL</name>
    <value>use_a group_a</value>
</property>
```
5. Now only user_a and other users in group_a can use the file in the mykey’s encryption zone.view encrypted zone:
```bash
hdfs  crypto -listZones
```

## Using BigDL Orca pytorch and tensorflow in SGX
Introduction to [BigDL-Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html).
Using a new image to test,and Occlum instance is already built in it. image_name:intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production-customer:orca-test-build.
```bash
#test.sh
bash /opt/mount.sh
occlum run /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -Djdk.lang.Process.launchMechanism=vfork \
                -cp "/opt/spark/conf/:/opt/spark/jars/*:/bin/jars/*" \
                -Xmx3g org.apache.spark.deploy.SparkSubmit \
                --conf "spark.pyspark.python=/bin/python3" \
                /py-examples/$1
```
### [Orca-pytorch quickstart](https://bigdl.readthedocs.io/en/latest/doc/Orca/Howto/pytorch-quickstart.html)
1.Set PROXY and enter the container according to the previous steps

2.Runnig Orca-pytorch quickstart example
```bash
cd /opt/occlum_spark && bash test.sh orca-pytorch.py
```

### [Orca-tensorflow quickstart](https://bigdl.readthedocs.io/en/latest/doc/Orca/Howto/tf2keras-quickstart.html)
1.Set PROXY and Enter the container according to the previous steps

2.Runnig Orca-tensorflow quickstart example
```bash
cd /opt/occlum_spark && bash test.sh orca-tf.py
```
## Using BigDL PPML Occlum EHSM Attestation Server
Bigdl ppml use EHSM as reference KMS&AS, you can deploy EHSM following the [guide](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#deploy-bigdl-ehsm-kms-on-kubernetes-with-helm-charts)
We assume you have already set up environment and enroll yourself on EHSM.

In [start-spark-local.sh](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/scala/docker-occlum/start-spark-local.sh). Set `ATTESTATION` = true and modify `PCCL_URL`, `ATTESTATION_URL` to the env value you have set,
and modify `APP_ID`, `API_KEY` to the value you have get  when enroll, and then you can change `CHALLENGE` and
`REPORT_DATA` for attestation.

``` bash
#start-spark-local.sh
-e ATTESTATION=false \   set to true to start attestation.
-e PCCS_URL=https://PCCS_IP:PCCS_PORT \  PCCS URL, obtained from KMS services or a self-deployed one. Should match the format https://<ip_address>:<port>.
-e ATTESTATION_URL=ESHM_IP:EHSM_PORT \  URL of attestation service. Should match the format <ip_address>:<port>.
-e APP_ID=your_app_id \ The appId generated by your attestation service.
-e API_KEY=your_api_key \ The apiKey generated by your attestation service.
-e CHALLENGE=cHBtbAo= \ Challenge is optional. Challenge is to get quote of attestation service which will be verified by local SGX SDK. Should be a BASE64 string. It can be a casual BASE64 string, for example, it can be generated by the command echo ppml|base64.
-e REPORT_DATA=ppml \ A random String to generator a quote which will be send to attestation service and use for attest. Default is ppml.
```
