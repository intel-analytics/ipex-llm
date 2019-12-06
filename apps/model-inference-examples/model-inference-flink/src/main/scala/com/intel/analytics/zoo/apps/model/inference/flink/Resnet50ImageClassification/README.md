## Analytics-Zoo InferenceModel with OpenVINO accelerating on Flink Streaming
This is the real-time image classification on Flink streaming. This example is loading pre-trained ResNet-50 model as Analytics Zoo Inference Model for image prediction on Flink. 

### Requirements

- JDK 1.8
- Flink 1.8.1
- scala 2.11/2.12
- Python 3.x
- Python libraries:

Install python dependencies for each Flink node. Notice the newest version of library may not be available.

```bash
sudo apt install python3-pip
pip3 install numpy networkx tensorflow
```

### Data and Model Preparation

- Prepare the prediction dataset from [imagenet](http://www.image-net.org/) and extract as you need.
- Prepare ResNet-50 model classes which can be extracted directly [here](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/resources/imagenet_classname.txt).
- Prepare saved model file for loading as OpenVINO IR. Download the pre-trained TensorFlow object detection model [resnet_v1_50 model](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz). Unzip it and extract the check point file.

### Configure and Start Flink

Edit /conf/flink-conf.yaml to set heap size or the number of task slots as you need, ie, `jobmanager.heap.size: 10g`, `taskmanager.numberOfTaskSlots: 2` 

Before running the example on Flink, you may make sure to start a flink cluster if there is no runing one:

```bash
./bin/start-cluster.sh
```

Check the Dispatcher's web frontend at [http://localhost:8081](http://localhost:8081/) and make sure everything is up and running. To stop Flink when you're done type:

```bash
./bin/stop-cluster.sh
```

### Run the Example
Go to the root directory of model-inference-flink and execute the `mvn clean package` command, which prepares the jar file for the model-inference-flink project. You can find the jar file in the new `./target` directory.

Run the following command with arguments to submit the Flink program. Change parameter settings as you need.

```bash
${FLINK_HOME}/bin/flink run \
    -m localhost:8081 -p 2 \
    -c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelType resnet_v1_50 --checkpointPath ${CHECKPOINT_PATH}  \
    --image ${IMAGE_PATH} --classes ${CLASSES_FILE} \
    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1
```
##### Options：

- `-m`  Address of the job manager to connect
- `-p` The number of task slots. Default is 2
- `-c` Class with the program entry point
- `--modelType` The pre-trained model type
- `--checkpointPath` The pre-trained ResNet-50 checkpoint file path
- `--image` Image folder path
- `--classes` Classes/labels file path
- `--inputShape` Input shape fed to an input node(s) of the model
- `--ifReverseInputChannels` Boolean value of if need reverse input channels. Switch the input channels order from RGB to BGR
- `--meanValues` and `--scale` Values used to perform normalization for inference model to convert to generated IR

### Results

Results may look like: 

```
Printing result ...
sweatshirt
Siberian husky
Program execution finished
Job with JobID 22f9c394f0df3bcb4135a1c7675daa7c has finished.
Job Runtime: 11285 ms
```
