# Quick Start Guide
This guide provides all supported quick start choices of Cluster Serving.

### Docker
Docker quick start is provided in [Programming Guide Quick Start](ProgrammingGuide.md#quick-start).

### IDE
Cluster Serving supports running in IDE, please refer to [ContributionGuide](ContributeGuide.md#debug-in-ide).

### Manually Install
#### Prerequisites
In terminal, install Redis, and export `REDIS_HOME`
```
$ export REDIS_VERSION=5.0.5
$ wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-${REDIS_VERSION}.tar.gz && \
    rm redis-${REDIS_VERSION}.tar.gz && \
    cd redis-${REDIS_VERSION} && \
    make
$ export REDIS_HOME=$(pwd)
```
install Flink, and export `FLINK_HOME`
```
$ export FLINK_VERSION=1.11.2
$ wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar xzf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz && 
    cd flink-${FLINK_VERSION}
$ export FLINK_HOME=$(pwd)
```
#### Start Cluster Serving
install Cluster Serving by
```
pip install analytics-zoo-serving
```
then in terminal call
```
$ cluster-serving-init
$ ls
```
you could see `zoo.jar` and `config.yaml` in directory

Prepare a model, if you do not have model, you can use Tensorflow official model provided in TFHub [Download link](https://tfhub.dev/tensorflow/resnet_50/classification/1?tf-hub-format=compressed)

Modify `config.yaml` to
```
modelPath: /path/to/model/dir
```

start Cluster Serving
```
$ cluster-serving-start
```
#### Check Status
After start Cluster Serving, use following command to check if job is ready
```
$FLINK_HOME/bin/flink list
```
you could see Cluster Serving job in output, if not, please look for help in [Debug Guide](DebugGuide.md)
and go to Redis shell to check if Redis is functioning
```
$REDIS_HOME/src/redis-cli
>> keys *
```
You should see `serving_stream` in output, this is the input queue of Cluster Serving

#### Predict
In python shell, prepare an image and run following,
```
from zoo.serving.client import *
input_api = InputQueue()
img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
data = cv2.imencode(".jpg", img)[1]
img_encoded = base64.b64encode(data).decode("utf-8")
input_api.enqueue("my-image", t={"b64": img_encoded})
```
In Redis shell, you could see result when showing all the keys like following if Cluster Serving finishes prediction. If nothing appears over 30 seconds, maybe something is wrong, please look for help in [Debug Guide](DebugGuide.md)
```
$REDIS_HOME/src/redis-cli
>> keys *
```
then use following to get output of Cluster Serving in python shell
```
output_api = OutputQueue()
result_ndarray = output_api.query("my-image")
```
