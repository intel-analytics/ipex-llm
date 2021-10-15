# BigDL Cluster Serving Programming Guide

## Installation
It is recommended to install Cluster Serving by pulling the pre-built Docker image to your local node, which have packaged all the required dependencies. Alternatively, you may also manually install Cluster Serving (through either pip or direct downloading), Redis on the local node.
#### Docker
```
docker pull intelanalytics/bigdl-cluster-serving
```
then, (or directly run `docker run`, it will pull the image if it does not exist)
```
docker run --name cluster-serving -itd --net=host intelanalytics/bigdl-cluster-serving:0.9.0
```
Log into the container
```
docker exec -it cluster-serving bash
```
`cd ./cluster-serving`, you can see all the environments prepared.

#### Manual installation

##### Requirements
Non-Docker users need to install [Flink 1.10.0+](https://archive.apache.org/dist/flink/flink-1.10.0/), 1.10.0 by default, [Redis 5.0.0+](https://redis.io/topics/quickstart), 5.0.5 by default.

For users do not have above dependencies, we provide following command to quickly set up.

Redis
```
$ export REDIS_VERSION=5.0.5
$ wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-${REDIS_VERSION}.tar.gz && \
    rm redis-${REDIS_VERSION}.tar.gz && \
    cd redis-${REDIS_VERSION} && \
    make
```

Flink
```
$ export FLINK_VERSION=1.11.2
$ wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar xzf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz.tgz
```

After preparing dependencies above, make sure the environment variable `$FLINK_HOME` (/path/to/flink-FLINK_VERSION-bin), `$REDIS_HOME`(/path/to/redis-REDIS_VERSION) is set before following steps. 

#### Install release version
```
pip install bigdl-serving
```
#### Install nightly version
Download package from [here](https://sourceforge.net/projects/bigdl/files/cluster-serving-py/), run following command to install Cluster Serving
```
pip install analytics_zoo_serving-*.whl
```
For users who need to deploy and start Cluster Serving, run `cluster-serving-init` to download and prepare dependencies.

For users who need to do inference, aka. predict data only, the environment is ready.

## Configuration
### Set up cluster
Cluster Serving uses Flink cluster, make sure you have it according to [Installation](#1-installation).

For docker user, the cluster should be already started. You could use `netstat -tnlp | grep 8081` to check if Flink REST port is working, if not, call `$FLINK_HOME/bin/start-cluster.sh` to start Flink cluster.

If you need to start Flink on yarn, refer to [Flink on Yarn](https://ci.apache.org/projects/flink/flink-docs-stable/deployment/resource-providers/yarn.html), or K8s, refer to [Flink on K8s](https://ci.apache.org/projects/flink/flink-docs-stable/deployment/resource-providers/standalone/kubernetes.html) at Flink official documentation.

If you use Flink standalone, call `$FLINK_HOME/bin/start-cluster.sh` to start Flink cluster.



### Configuration file
After [Installation](#1-installation), you will see a config file `config.yaml` in your current working directory. This file contains all the configurations that you can customize for your Cluster Serving. See an example of `config.yaml` below.
```
## BigDL Cluster Serving Config Example
# model path must be provided
modelPath: /path/to/model
```

### Preparing Model
Currently BigDL Cluster Serving supports TensorFlow, OpenVINO, PyTorch, BigDL, Caffe models. Supported types are listed below.

You need to put your model file into a directory with layout like following according to model type, note that only one model is allowed in your directory. Then, set in `config.yaml` file with `modelPath:/path/to/dir`.

**Tensorflow**
***Tensorflow SavedModel***
```
|-- model
   |-- saved_model.pb
   |-- variables
       |-- variables.data-00000-of-00001
       |-- variables.index
```
***Tensorflow Frozen Graph***
```
|-- model
   |-- frozen_inference_graph.pb
   |-- graph_meta.json
```
**note:** `.pb` is the weight file which name must be `frozen_inference_graph.pb`, `.json` is the inputs and outputs definition file which name must be `graph_meta.json`, with contents like `{"input_names":["input:0"],"output_names":["output:0"]}`

***Tensorflow Checkpoint***
Please refer to [freeze checkpoint example](https://github.com/intel-analytics/bigdl/tree/master/pyzoo/bigdl/examples/tensorflow/freeze_checkpoint)

**Pytorch**

```
|-- model
   |-- xx.pt
```
Running Pytorch model needs extra dependency and config. Refer to [here](https://github.com/intel-analytics/bigdl/blob/master/pyzoo/bigdl/examples/pytorch/train/README.md) to install dependencies, and set environment variable `$PYTHONHOME` to your python, e.g. python could be run by `$PYTHONHOME/bin/python` and library is at `$PYTHONHOME/lib/`.

**OpenVINO**

```
|-- model
   |-- xx.xml
   |-- xx.bin
```
**BigDL**

```
|--model
   |-- xx.model
```
**Caffe**

```
|-- model
   |-- xx.prototxt
   |-- xx.caffemodel
```


### Other Configuration
The field `params` contains your inference parameter configuration.

* core_number: the **batch size** you use for model inference, usually the core number of your machine is recommended. Thus you could just provide your machine core number at this field. We recommend this value to be not smaller than 4 and not larger than 512. In general, using larger batch size means higher throughput, but also increase the latency between batches accordingly.

### High Performance Configuration Recommended
#### Tensorflow, Pytorch
1 <= thread_per_model <= 8, in config
```
# default: number of models used in serving
# modelParallelism: core_number of your machine / thread_per_model
```
environment variable
```
export OMP_NUM_THREADS=thread_per_model
```
#### OpenVINO
environment variable
```
export OMP_NUM_THREADS=core_number of your machine
```
