## OpenVINO Object Detection example

This example illustrates how to use a pre-trained TensorFlow object detection model
to make inferences with OpenVINO toolkit as backend using Analytics Zoo, which delivers a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).


## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

[OpenVINO System requirements](https://software.intel.com/en-us/openvino-toolkit/documentation/system-requirements):

    Ubuntu 16.04.3 LTS (64 bit)
    Windows 10 (64 bit)
    CentOS 7.4 (64 bit)
    macOS 10.13, 10.14 (64 bit)

OpenVINO Python requirements:

    tensorflow>=1.2.0
    networkx>=1.11
    numpy>=1.12.0
    protobuf==3.6.1

## Model and Data Preparation
1. Prepare a pre-trained TensorFlow object detection model. You can download from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
   
In this example, we use `frozen_inference_graph.pb` of the `faster_rcnn_resnet101_coco` model downloaded from [here](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz).


2. Prepare the image dataset for inference. Put the images to do prediction in the same folder.


## Run this example after pip install
```bash
export SPARK_DRIVER_MEMORY=10g
image_path=directory path that contain images
model_path=dir contains frozen_inference_graph.pb and pipeline.config

python predict.py --image ${image_path} --model ${model_path}
```

See [here](#options) for more configurable options for this example.


## Run this example with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the directory where you extract the downloaded Analytics Zoo zip package
MASTER=local[*]
image_path=directory path that contain images
model_path=dir contains frozen_inference_graph.pb and pipeline.config

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master $MASTER \
    --driver-memory 10g \
    --executor-memory 10g \
    predict.py \
    --image ${image_path} \
    --model ${model_path}
```

See [here](#options) for more configurable options for this example.


## Options
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path to the TensorFlow object detection model.
* `--model_type` The type of the TensorFlow model. In this example, we use "faster_rcnn_resnet101_coco". If you download other models, you need to change to the corresponding model_type.

## Results
We print the detection result of the first image to the console.
