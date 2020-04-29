## Image Classification Streaming with Analytics Zoo Inference Model on Flink
This is the real-time image classification on Flink streaming. This example is loading pre-trained TensorFlow MobileNet model as Analytics Zoo TFNet for image prediction on Flink. 

### Requirements

- JDK 1.8
- Flink 1.8.1
- scala 2.11/2.12

### Data and Model Preparation

- Prepare the prediction images from [imagenet](http://www.image-net.org/) and extract as you need.
- Prepare ImageNet classes which can be extracted directly [here](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/resources/imagenet_classname.txt).
- Prepare MobileNet_V1 frozen model file for loading as TFNet. Download the pre-trained [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz). Unzip it and extract the `mobilenet_v1_1.0_224_frozen.pb file`.

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
    -c com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification.ImageClassificationStreaming  \
    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
    --modelPath ${MODEL_PATH} --modelType "frozenModel"  \
    --images ${IMAGE_PATH} --classes ${CLASSES_FILE} \
    --modelInputs "input:0"  --modelOutputs "MobilenetV1/Predictions/Reshape_1:0"
    --intraOpParallelismThreads 1 --interOpParallelismThreads 1 --usePerSessionThreads true
    --output ${OUTPUT}
```
##### Options：

- `-m` Address of the job manager to connect
- `-p` The number of task slots. Default is 2
- `-c` Class with the program entry point
- `--modelPath` The pre-trained model file path. It should be `./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb`
- `--modelType` The pre-trained model file format. This is frozenModel
- `--images` Image folder path
- `--classes` ImageNet classes/labels file path
- `--modelInputs` Input node(s) of the model
- `--modelOutputs` Output node(s) of the model. The information about input and output nodes of the model can be found in the downloaded tar file at `./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_info.txt`
- `--intraOpParallelismThreads` The num of intraOpParallelismThreads. Default is 1
- `--interOpParallelismThreads` The num of interOpParallelismThreads. Default is 1
- `--usePerSessionThreads` Boolean value whether to perSessionThreads. Default is true
- `--output` File path of predication results 

### Results

Results will be the class/label name. It may look like: 

```
sweatshirt
Siberian husky
```
