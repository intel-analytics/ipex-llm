## Analytics-Zoo InferenceModel with OpenVINO accelerating on Flink Streaming
This is the example of streaming with Flink and Resnet50 model, as well as using Analytics-Zoo InferenceModel to accelerate prediction.

### Requirements
* JDK 1.8
* Flink 1.8.1
* scala 2.11/2.12
* Python 3.x

### Environment
Install dependencies for each flink node.
```
sudo apt install python3-pip
pip3 install numpy
pip3 install networkx
pip3 install tensorflow
```
### Start and stop Flink
you may start a flink cluster if there is no runing one:
```
./bin/start-cluster.sh
```
Check the Dispatcher's web frontend at http://localhost:8081 and make sure everything is up and running.
To stop Flink when you're done type:
```
./bin/stop-cluster.sh
```

### Run the Example
* Run `export FLINK_HOME=the root directory of flink`.
* Run `export ANALYTICS_ZOO_HOME=the folder of Analytics Zoo project`.
* Download [resnet_v1_50 model](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). Run `export MODEL_PATH=path to the downloaded model`.
* Prepare the prediction dataset from [imagenet](http://www.image-net.org/) and extract as you need.
* Go to the root directory of model-inference-flink and execute the `mvn clean package` command, which prepares the jar file for model-inference-flink.
* Edit flink-conf.yaml to set heap size or the number of task slots as you need, ie,  `jobmanager.heap.size: 10g`
* Run the follwing command with arguments to submit the Flink program. Change parameter settings as you need.

```bash
${FLINK_HOME}/bin/flink run \
    -m localhost:8081 -p ${task_slot_num} \
    -c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelType resnet_v1_50 --checkpointPath ${MODEL_PATH}  \
    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1
```
