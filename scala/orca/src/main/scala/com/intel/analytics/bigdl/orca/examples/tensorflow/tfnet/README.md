## TFNet Object Detection example

TFNet can encapsulate a frozen TensorFlow graph as an Analytics Zoo layer for inference.

This example illustrates how to use a pre-trained TensorFlow object detection model
to make inferences using Analytics Zoo on Spark.

## Install Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) to install analytics-zoo.


## Model and Data Preparation
1. Prepare a pre-trained TensorFlow object detection model. You can download from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

In this example, we use `frozen_inference_graph.pb` of the `ssd_mobilenet_v1_coco` model downloaded from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz).

2. Prepare the image dataset for inference. Put the images to do prediction in the same folder.

## Run this example
Run the following command for Spark local mode (`master=local[*]`) or cluster mode:

```bash
master=... // spark master
modelPath=... // model path.
imagePath=... // image path.

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tensorflow.tfnet.Predict \
--image $imagePath --model $modelPath --partition 4
```

__Options:__
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path of the TensorFlow object detection model. Local file system, HDFS and Amazon S3 are supported.
* `--partition` The number of partitions to cut the dataset into. Default is 4.

## Results
The result of this example will be the detection results of the input images, with the first detection box of an image having the highest prediction score.

We print the first prediction result to the console.
