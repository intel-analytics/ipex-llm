## Overview

This is a Scala example for image inference with a pre-trained caffe Inception_V1 model based
on Spark DataFrame (Dataset).

Analytics Zoo provides the DataFrame-based API for image reading, pre-processing.
The related classes followed the typical estimator/transformer pattern of Spark ML and can be used in
a standard Spark ML pipeline.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Image Model Inference

You can run ModelInference example by the following steps.

1. Prepare pre-trained model and definition file.
Download caffe inception v1 [weights](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
and [deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt).
Put the files in `/tmp/zoo` or other path.

2. Prepare predict dataset
You can use your own image data (JPG or PNG), or some images from imagenet-2012 validation
dataset <http://image-net.org/download-images> to run the example. we use `/tmp/zoo/infer_images`
for this example.

3. Run this example

Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode, adjust
 the memory size according to your image:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[1] \
    --driver-memory 3g \
    --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
    --caffeDefPath /tmp/zoo/deploy.prototxt \
    --caffeWeightsPath /tmp/zoo/bvlc_googlenet.caffemodel \
    --batchSize 32 \
    --imagePath /tmp/zoo/infer_images
```


After inference, you should see something like this in the console:
```
+-------------------------------------------------------+----------+
|imageName                                              |prediction|
+-------------------------------------------------------+----------+
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000001.JPEG|59.0      |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000002.JPEG|796.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000003.JPEG|231.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000004.JPEG|970.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000005.JPEG|432.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000006.JPEG|59.0      |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000007.JPEG|378.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000008.JPEG|713.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000009.JPEG|107.0     |
|file:/tmp/zoo/infer_images/ILSVRC2012_val_00000010.JPEG|284.0     |
+-------------------------------------------------------+----------+
only showing top 10 rows

```

To map the class to human readable text, please refer to https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt

In this example, we use the Inception-V1 model. Please feel free to explore other models from
Caffe, Keras and Tensorflow. Analytics Zoo provides popular pre-trained model in https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/#download-link
