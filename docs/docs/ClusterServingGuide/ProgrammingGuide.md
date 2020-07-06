# Programming Guide
Analytics Zoo Cluster Serving is a lightweight distributed, real-time serving solution that supports a wide range of deep learning models (such as TensorFlow, PyTorch, Caffe, BigDL and OpenVINO models). It provides a simple pub/sub API, so that the users can easily send their inference requests to the input queue (using a simple Python API); Cluster Serving will then automatically manage the scale-out and real-time model inference across a large cluster (using distributed streaming frameworks such as Apache Spark Streaming, Apache Flink, etc.) 

The overall architecture of Analytics Zoo Cluster Serving solution is illustrated as below: 

![overview](cluster_serving_overview.jpg)


This page contains the guide for you to run Analytics Zoo Cluster Serving, including following:

* [Quick Start](#quick-start)

* [Workflow Overview](#workflow-overview) 

* [Deploy Your Own Cluster Serving](#deploy-your-own-cluster-serving)

   1. [Installation](#1-installation)

   2. [Configuration](#2-configuration) 
   
   3. [Launching Service](#3-launching-service)
   
   4. [Model inference](#4-model-inference)

   5. [HTTP Server](#5-http-server)

* [Optional Operations](#optional-operations)

     - [Update Model or Configurations](#update-model-or-configurations)

     - [Logs and Visualization](#logs-and-visualization)


## Quick Start

This section provides a quick start example for you to run Analytics Zoo Cluster Serving. To simplify the example, we use docker to run Cluster Serving. If you do not have docker installed, [install docker](https://docs.docker.com/install/) first. The quick start example contains all the necessary components so the first time users can get it up and running within minutes:

* A docker image for Analytics Zoo Cluster Serving (with all dependencies installed)
* A sample configuration file
* A sample trained TensorFlow model, and sample data for inference
* A sample Python client program

Use one command to run Cluster Serving container. (We provide quick start model in older version of docker image, for newest version, please refer to following sections and we remove the model to reduce the docker image size).
```
docker run -itd --name cluster-serving --net=host intelanalytics/zoo-cluster-serving:0.7.0
```
Log into the container using `docker exec -it cluster-serving bash`. 

We already prepared `analytics-zoo` and `opencv-python` with pip in this container. And prepared model in `model` directory with following structure.

```
cluster-serving | 
               -- | model
                 -- frozen_graph.pb
                 -- graph_meta.json
```

Start Cluster Serving using `cluster-serving-start`. 

Run python program `python quick_start.py` to push data into queue and get inference result. 

Then you can see the inference output in console. 
```
image: fish1.jpeg, classification-result:class: 5's prob: 0.18204997
image: dog1.jpeg, classification-result:class: 267's prob: 0.27166227
image: cat1.jpeg, classification-result:class: 292's prob: 0.32633427
```
Wow! You made it!

Note that the Cluster Serving quick start example will run on your local node only. Check the [Deploy Your Own Cluster Serving](#deploy-your-own-cluster-serving) section for how to configure and run Cluster Serving in a distributed fashion.

For more details, you could also see the log and performance by go to `localhost:6006` in your browser and refer to [Logs and Visualization](#logs-and-visualization), or view the source code of `quick_start.py` [here](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/serving/quick_start.py), or refer to [API Guide](APIGuide.md).


## Workflow Overview

The figure below illustrates the simple 3-step "Prepare-Launch-Inference" workflow for Cluster Serving.

![steps](cluster_serving_steps.jpg)

#### 1. Install and prepare Cluster Serving environment on a local node:

- Copy a previously trained model to the local node; currently TensorFlow, PyTorch, Caffe, BigDL and OpenVINO models are supported.
- Install Analytics Zoo on the local node (e.g., using a single pip install command)
- Configure Cluster Server on the local node, including the file path to the trained model and the address of the cluster (such as Apache Hadoop YARN cluster, Spark Cluster, K8s cluster, etc.).
Please note that you only need to deploy the Cluster Serving solution on a single local node, and NO modifications are needed for the (YARN or K8s) cluster. 

#### 2. Launch the Cluster Serving service

You can launch the Cluster Serving service by running the startup script on the local node. Under the hood, Cluster Serving will automatically deploy the trained model and serve the model inference requests across the cluster in a distributed fashion. You may monitor its runtime status (such as inference throughput) using TensorBoard. 

#### 3. Distributed, real-time (streaming) inference

Cluster Serving provides a simple pub/sub API to the users, so that you can easily send the inference requests to an input queue (currently Redis Streams is used) using a simple Python API.

Cluster Serving will then read the requests from the Redis stream, run the distributed real-time inference across the cluster (using Spark Streaming or Flink), and return the results back through Redis. As a result, you may get the inference results again using a simple Python API.


## Deploy your Own Cluster Serving
### 1. Installation
It is recommended to install Cluster Serving by pulling the pre-built Docker image to your local node, which have packaged all the required dependencies. Alternatively, you may also manually install Cluster Serving (through either pip or direct downloading) as well as Spark, Redis and TensorBoard (for visualizing the serving status) on the local node.
#### Docker
```
docker pull intelanalytics/zoo-cluster-serving
```
then, (or directly run `docker run`, it will pull the image if it does not exist)
```
docker run --name cluster-serving --net=host -itd intelanalytics/zoo-cluster-serving:0.8.1 bash
```
Log into the container
```
docker exec -it cluster-serving bash
```
`cd ./cluster-serving`, you can see all the environments are prepared.
##### Yarn user
For Yarn user using docker, you have to set additional config, thus you need to call following when starting the container
```
docker run --name cluster-serving --net=host -v /path/to/HADOOP_CONF_DIR:/opt/work/HADOOP_CONF_DIR -e HADOOP_CONF_DIR=/opt/work/HADOOP_CONF_DIR -itd intelanalytics/zoo-cluster-serving:0.8.1 bash
```

#### Manual installation
Non-Docker users need to install [Flink](https://archive.apache.org/dist/flink/flink-1.10.0/), 1.10.0 by default, for users choose Spark as backend, install [Spark](https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz), 2.4.3 by default, [Redis](https://redis.io/topics/quickstart), 0.5.0 by default and [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) if you choose Spark backend and need visualization.

After preparing dependencies above, make sure the environment variable `$FLINK_HOME` (/path/to/flink-FLINK_VERSION-bin), or `$SPARK_HOME` (/path/to/spark-SPARK_VERSION-bin-hadoop-HADOOP_VERSION), `$REDIS_HOME`(/path/to/redis-REDIS_VERSION) is set before following steps. 

For Spark user only, use `pip install tensorboard` to install TensorBoard.

Install Analytics Zoo by Download Release or Pip.

##### Download Release
Download Analytics Zoo from [release page](https://analytics-zoo.github.io/master/#release-download/) on the local node. 

Unzip the file and go into directory `analytics-zoo`, run `export ANALYTICS_ZOO_HOME=$(pwd)` to set `$ANALYTICS_ZOO_HOME` variable.

Run `source analytics-zoo/bin/analytics-zoo-env.sh` to set environment.

Go to `analytics-zoo/bin/cluster-serving`, run `cluster-serving-init`.

Run `export OMP_NUM_THREADS=all` if you want to use all cores on your machine to do inference in parallel manner.
##### Pip
Use `pip install analytics-zoo` to install release stable version. For latest nightly built version, download the wheel at [download page](https://sourceforge.net/projects/analytics-zoo/files/zoo-py/) and use `pip` to install it.

Then, go to any directory, run `cluster-serving-init`.

Run `export OMP_NUM_THREADS=all` if you want to use all cores on your machine to do inference in parallel manner.
### 2. Configuration
#### How to Config
After [Installation](#1-installation), you will see a config file `config.yaml` in your current working directory. This file contains all the configurations that you can customize for your Cluster Serving. See an example of `config.yaml` below.
```
## Analytics Zoo Cluster Serving Config Example

model:
  # model path must be set
  path: /opt/work/model
  # the inputs of the tensorflow model, separated by ","
  inputs:
  # the outputs of the tensorflow model, separated by ","
  outputs:
data:
  # default, localhost:6379
  src:
  # default, image (image & tensor are supported)
  data_type: 
  # default, 3,224,224
  image_shape:
  # must be provided given data_type is tensor. eg: [1,2] (tensor) [[1],[2,1,2],[3]] (table)
  tensor_shape: 
  # default, topN(1)
  filter:
params:
  # default, 4
  batch_size:
  # default: OFF
  performance_mode:
spark:
  # default, local[*], change this to spark://, yarn, k8s:// etc if you want to run on cluster
  master: local[*]
  # default, 4g
  driver_memory:
  # default, 1g
  executor_memory:
  # default, 1
  num_executors:
  # default, 4
  executor_cores:
  # default, 4
  total_executor_cores:
```

#### Preparing Model
Currently Analytics Zoo Cluster Serving supports TensorFlow, OpenVINO, PyTorch, BigDL, Caffe models. Supported types are listed below.

You need to put your model file into a directory and the directory could have layout like following according to model type, note that only one model is allowed in your directory.

**Tensorflow**

***Tensorflow checkpoint***
Please refer to [freeze checkpoint example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/tensorflow/freeze_checkpoint)

***Tensorflow frozen model***
```
|-- model
   |-- frozen_graph.pb
   |-- graph_meta.json
```

***Tensorflow saved model***
```
|-- model
   |-- saved_model.pb
   |-- variables
       |-- variables.data-00000-of-00001
       |-- variables.index
```
Note: you can specify model inputs and outputs in the config.yaml file. If the inputs or outputs are not provided, the signature "serving_default" will be used to find input and output tensors.

**Caffe**

```
|-- model
   |-- xx.prototxt
   |-- xx.caffemodel
```

**Pytorch**

```
|-- model
   |-- xx.pt
```

**BigDL**

```
|--model
   |-- xx.model
```

**OpenVINO**

```
|-- model
   |-- xx.xml
   |-- xx.bin
```

Put the model in any of your local directory, and set `model:/path/to/dir`.

#### Data Configuration
The field `data` contains your input data configuration.

* src: the queue you subscribe for your input data, e.g. default config of Redis on local machine is `localhost:6379`. Note that please use the host address in your network instead of localhost or 127.0.0.1 when you run serving in cluster, and make sure other nodes in cluster could also recognize this address.
* shape: the shape of your input data. e.g. [[1],[3,224,224],[3]], if your model contains only one input, brackets could be omitted.
* filter: the post-processing of pipeline, could be none. Except none, currently supported filters are,

Top-N, e.g. `topN(1)` represents Top-1 result is kept and returned with index. User should follow this schema `topN(n)`. Noted if the top-N number is larger than model output size of the the final layer, it would just return all the outputs.

#### Other Configuration
The field `params` contains your inference parameter configuration.

* core_number: the batch size you use for model inference, usually the core number of your machine is recommended. Thus you could just provide your machine core number at this field. We recommend this value to be not smaller than 4 and not larger than 512. In general, using larger batch size means higher throughput, but also increase the latency between batches accordingly.
* performance_mode: The performance mode will utilize your CPU resource to achieve better inference performance on a single node. **Note:** numactl and net-tools should be installed in your system, and spark master should be `local[*]` in the config.yaml file.

For Spark users only, the field `spark` contains your spark configuration.

* master: Your cluster address, same as parameter `master` in spark
* driver_memory: same as parameter `driver-memory` in spark
* executor_memory: same as parameter `executor-memory` in spark
* num_executors: same as parameter `num-executors` in spark
* executor_cores: same as paramter `executor-cores` in spark
* total_executor_cores: same as parameter ` total-executor-cores` in spark

For more details of these config, please refer to [Spark Official Document](https://spark.apache.org/docs/latest/configuration.html)
### 3. Launching Service
We provide following scripts to start, stop, restart Cluster Serving. 
#### Start
You can use following command to start Cluster Serving.
```
cluster-serving-start
```
This command will start Redis and TensorBoard (for spark users only) if they are not running.

For spark users, if you choose spark streaming, run `spark-streaming-cluster-serving-start`. If you choose spark structured streaming, run `spark-structured-streaming-cluster-serving-start`.

#### Stop
You can use following command to stop Cluster Serving. Data in Redis and TensorBoard service will persist.
```
cluster-serving-stop
```
#### Restart
You can use following command to restart Cluster Serving.
```
cluster-serving-restart
```
#### Shut Down
You can use following command to shutdown Cluster Serving. This operation will stop all running services related to Cluster Serving, specifically, Redis and TensorBoard. Note that your data in Redis will be removed when you shutdown. 
```
cluster-serving-shutdown
```

If you are using Docker, you could also run `docker rm` to shutdown Cluster Serving.
### 4. Model Inference
We support Python API and HTTP RESTful API for conducting inference with Data Pipeline in Cluster Serving. 

#### Python API
For Python API, the requirements of python packages are `opencv-python`(for raw image only), `pyyaml`, `redis`. You can use `InputQueue` and `OutputQueue` to connect to data pipeline by providing the pipeline url, e.g. `my_input_queue = InputQueue(host, port)` and `my_output_queue = OutputQueue(host, port)`. If parameters are not provided, default url `localhost:6379` would be used.

We provide some basic usages here, for more details, please see [API Guide](APIGuide.md).
##### Input and Output API
To input data to queue, you need a `InputQueue` instance, and using `enqueue` method, for each input, give a key correspond to your model or give arbitrary key if your model does not care about it.

To enqueue an image
```
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```
To enqueue an instance containing 1 image and 2 ndarray
```
from zoo.serving.client import InputQueue
import numpy as np
input_api = InputQueue()
t1 = np.array([1,2])
t2 = np.array([[1,2], [3,4]])
input_api.enqueue('my-instance', img={"path": 'path/to/image'}, tensor1=t1, tensor2=t2)
```
There are 4 types of inputs in total, string, image, tensor, sparse tensor, which could represents nearly all types of models. For more details of usage, go to [API Guide](APIGuide.md)

To get data from queue, you need a `OutputQueue` instance, and using `query` or `dequeue` method. The `query` method takes image uri as parameter and returns the corresponding result. The `dequeue` method takes no parameter and just returns all results and also delete them in data queue. See following example.
```
from zoo.serving.client import OutputQueue
output_api = OutputQueue()
img1_result = output_api.query('img1')
all_result = output_api.dequeue() # the output queue is empty after this code
```
##### Output Format
Consider the code above, in [Input and Output API](#input-and-output-api) Section.
```
img1_result = output_api.query('img1')
```
The `img1_result` is a json format string, like following:
```
'{"class_1":"prob_1","class_2":"prob_2",...,"class_n","prob_n"}'
```
Where `n` is the number of `top_n` in your configuration file. This string could be parsed by `json.loads`.
```
import json
result_class_prob_map = json.loads(img1_result)
```

#### HTTP RESTful API
For HTTP RESTful API, we provide a HTTP server to support RESTful HTTP requests. User can submit HTTP requests to the HTTP server through RESTful APIs. The HTTP server will parse the input requests and pub them to Redis input queues, and also retrieve the output results and render them as json results in HTTP responses. The serving backend will leverage the cluster serving.

##### Start the HTTP Server
User can download a analytics-zoo-${VERSION}-http.jar from the Nexus Repository with GAVP: 
```
<groupId>com.intel.analytics.zoo</groupId>
<artifactId>analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}</artifactId>
<version>${ZOO_VERSION}</version>
```
User can also build from the source code:
```
mvn clean package -P spark_2.4+ -Dmaven.test.skip=true
```
After that, start the HTTP server with below command.
```
java -jar analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ZOO_VERSION}-http.jar
```
And check the status of the HTTP server with:
```
curl  http://${BINDED_HOST_IP}:${BINDED_HOST_PORT}/
```
If you get a response like "welcome to analytics zoo web serving frontend", that means the HTTP server is started successfully.
##### Start options
User can pass options to the HTTP server when start it:
```
java -jar analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ZOO_VERSION}-http.jar --redisHost="172.16.0.109"
```
All the supported parameter are listed here:
* **interface**: the binded server interface, default is "0.0.0.0"
* **port**: the binded server port, default is 10020
* **redisHost**: the host IP of redis server, default is "localhost"
* **redisPort**: the host port of redis server, default is 6379
* **redisInputQueue**: the input queue of redis server, default is "serving_stream"
* **redisOutputQueue**: the output queue of redis server, default is "result:" 
* **parallelism**: the parallelism of requests processing, default is 1000
* **timeWindow**: the timeWindow wait to pub inputs to redis, default is 0
* **countWindow**: the timeWindow wait to ub inputs to redis, default is 56
* **tokenBucketEnabled**: the switch to enable/disable RateLimiter, default is false
* **tokensPerSecond**: the rate of permits per second, default is 100
* **tokenAcquireTimeout**: acquires a permit from this RateLimiter if it can be obtained without exceeding the specified timeout(ms), default is 100

**User can adjust these options to tune the performance of the HTTP server.**

##### RESTful API
This part describes API endpoints and end-to-end examples on usage. 
The requests and responses are in JSON format. The composition of them depends on the requests type or verb. See the APIs for details.
In case of error, all APIs will return a JSON object in the response body with error as key and the error message as the value:
```
{
  "error": <error message string>
}
```
##### Predict API
URL
```
POST http://host:port/predict
```
Request Example for images as inputs:
```
curl -d \
'{
  "instances": [
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    },
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    },
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    },
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    },
    {
      "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
    }
  ]
}' \
-X POST http://host:port/predict
```
Response Example
```
{
  "predictions": [
    "{value=[[903,0.1306194]]}",
    "{value=[[903,0.1306194]]}",
    "{value=[[903,0.1306194]]}",
    "{value=[[903,0.1306194]]}",
    "{value=[[903,0.1306194]]}"
  ]
}
```
Request Example for tensor as inputs:
```
curl -d \
'{
  "instances" : [ {
    "ids" : [ 100.0, 88.0 ]
  }, {
    "ids" : [ 100.0, 88.0 ]
  }, {
    "ids" : [ 100.0, 88.0 ]
  }, {
    "ids" : [ 100.0, 88.0 ]
  }, {
    "ids" : [ 100.0, 88.0 ]
  } ]
}' \
-X POST http://host:port/predict
```
Response Example
```
{
  "predictions": [
    "{value=[[1,0.6427843]]}",
    "{value=[[1,0.6427843]]}",
    "{value=[[1,0.6427843]]}",
    "{value=[[1,0.6427843]]}",
    "{value=[[1,0.6427842]]}"
  ]
}
```
Another request example for composition of scalars and tensors.
```
curl -d \
 '{
  "instances" : [ {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, arrow.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897, 0.32762378, 0.45966738, 0.30405098, 0.62053126, 0.7037923 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ], [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ]
  }, {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, arrow.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897, 0.32762378, 0.45966738, 0.30405098, 0.62053126, 0.7037923 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ], [ [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ], [ [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ]
  } ]
}' \
-X POST http://host:port/predict
```
Another request example for composition of sparse and dense tensors.
```
curl -d \
'{
  "instances" : [ {
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
  }, {
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ]
  } ]
}' \
-X POST http://host:port/predict
```


##### Metrics API
URL
```
GET http://host:port/metrics
```
Response example:
```
[
  {
    name: "zoo.serving.redis.get",
    count: 810,
    meanRate: 12.627772820651845,
    min: 0,
    max: 25,
    mean: 0.9687099303718213,
    median: 0.928579,
    stdDev: 0.8150031623593447,
    _75thPercentile: 1.000047,
    _95thPercentile: 1.141443,
    _98thPercentile: 1.268665,
    _99thPercentile: 1.608387,
    _999thPercentile: 25.874584
  },
  {
    name: "zoo.serving.redis.put",
    count: 192,
    meanRate: 2.9928448518681816,
    min: 4,
    max: 207,
    mean: 8.470988823179553,
    median: 6.909573,
    stdDev: 13.269285415774808,
    _75thPercentile: 8.262833,
    _95thPercentile: 14.828704,
    _98thPercentile: 18.860232,
    _99thPercentile: 19.825203,
    _999thPercentile: 207.541874
  },
  {
    name: "zoo.serving.redis.wait",
    count: 192,
    meanRate: 2.992786169232195,
    min: 82,
    max: 773,
    mean: 93.03099107296806,
    median: 88.952799,
    stdDev: 45.54085374821418,
    _75thPercentile: 91.893393,
    _95thPercentile: 118.370628,
    _98thPercentile: 119.941905,
    _99thPercentile: 121.158649,
    _999thPercentile: 773.497556
  },
  {
    name: "zoo.serving.request.metrics",
    count: 1,
    meanRate: 0.015586927261874562,
    min: 18,
    max: 18,
    mean: 18.232472,
    median: 18.232472,
    stdDev: 0,
    _75thPercentile: 18.232472,
    _95thPercentile: 18.232472,
    _98thPercentile: 18.232472,
    _99thPercentile: 18.232472,
    _999thPercentile: 18.232472
  },
  {
    name: "zoo.serving.request.overall",
    count: 385,
    meanRate: 6.000929977336221,
    min: 18,
    max: 894,
    mean: 94.5795886310155,
    median: 89.946348,
    stdDev: 49.63620144068503,
    _75thPercentile: 93.851032,
    _95thPercentile: 121.148026,
    _98thPercentile: 123.118267,
    _99thPercentile: 124.053326,
    _999thPercentile: 894.004612
  },
  {
    name: "zoo.serving.request.predict",
    count: 192,
    meanRate: 2.9925722215434205,
    min: 85,
    max: 894,
    mean: 96.63308151066575,
    median: 92.323305,
    stdDev: 53.17110030594844,
    _75thPercentile: 94.839714,
    _95thPercentile: 122.564496,
    _98thPercentile: 123.974892,
    _99thPercentile: 125.636335,
    _999thPercentile: 894.062819
  }
]
```

## Optional Operations
### Update Model or Configurations
To update your model, you could replace your model file in your model directory, and restart Cluster Serving by `cluster-serving-restart`. Note that you could also change your configurations in `config.yaml` and restart serving.

### Logs and Visualization
#### Logs
We use log to save Cluster Serving information and error. To see log, please refer to `cluster-serving.log`.

#### Visualization
To visualize Cluster Serving performance, go to your flink job UI, default `localhost:8081`, and go to Cluster Serving job -> metrics. Add `numRecordsOut` to see total record number and `numRecordsOutPerSecond` to see throughput.

See example of visualization:

![Example Chart](serving-visualization.png)

##### Spark Streaming Visualization
TensorBoard is integrated into Spark Streaming Cluster Serving. TensorBoard service is started with Cluster Serving. Once your serving starts, you can go to `localhost:6006` to see visualization of your serving.

Analytics Zoo Cluster Serving provides 2 attributes in Tensorboard so far, `Serving Throughput` and `Total Records Number`.

* `Serving Throughput`: The overall throughput, including preprocessing and postprocessing of your serving. The line should be relatively stable after first few records. If this number has a drop and remains lower than previous, you might have lost the connection of some nodes in your cluster.

* `Total Records Number`: The total number of records that Cluster Serving gets so far.

See example of visualization:

![Example Chart](example-chart.png)
