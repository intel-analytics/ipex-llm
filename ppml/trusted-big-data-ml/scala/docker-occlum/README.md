# Trusted Big Data ML with Occlum


## Prerequisites

Pull image from dockerhub.

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT
```

Also, you can build image with `build-docker-image.sh`. Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:

``` bash
bash build-docker-image.sh
```

## Before run example
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

You can see Pi result in logs (`docker attach logs -f bigdl-ppml-trusted-big-data-ml-scala-occlum`)

```bash
Pi is roughly 3.1436957184785923
```

## BigDL Lenet Mnist Example

To train a model with PPML in BigDL, you need to prepare the data first. You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). There are 5 files in total. `train-images-idx3-ubyte` contains train images; `train-labels-idx1-ubyte` is the train label file; `t10k-images-idx3-ubyte` has validation images; `t10k-labels-idx1-ubyte` contains validation labels. Unzip all the files and put them in a new directory `data`.

**By default, `data` dir will be mounted to `/opt/occlum_spark/data` in container (become `/host/data` in occlum). You can change data path in `start-spark-local.sh`.**

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
