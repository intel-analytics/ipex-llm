## TFNet Object Detection example

TFNet can encapsulate a frozen TensorFlow graph as an Analytics Zoo layer for inference.

This example illustrates how to use a pre-trained TensorFlow object detection model
to make inferences using Analytics Zoo on Spark.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model and Data Preparation
1. Prepare a pre-trained TensorFlow object detection model. You can download from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

In this example, we use `frozen_inference_graph.pb` of the `ssd_mobilenet_v1_coco` model downloaded from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz).

2. Prepare the image dataset for inference. Put the images to do prediction in the same folder.


## Run this example after pip install
```bash
python predict.py --image path_to_image_folder --model path_to_tensorflow_graph
```

__Options:__
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path to the TensorFlow object detection model. Local file system, HDFS and Amazon S3 are supported.
* `--partition_num` The number of partitions to cut the dataset into. Default is 4.

## Run this example with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the directory where you extract the downloaded Analytics Zoo zip package
MASTER=... # Spark Master
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    predict.py \
    --image path_to_image_folder \
    --model path_to_tensorflow_graph
```

__Options:__
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path of the TensorFlow object detection model.
* `--partition_num` The number of partitions to cut the dataset into. Default is 4.

## Results
The result of this example will be the detection boxes (y_min, x_min, y_max, x_max) of the input images, with the first detection box of an image having the highest prediction score.

We print the detection box with the highest score of the first prediction result to the console.
