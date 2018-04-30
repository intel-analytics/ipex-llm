# Overview

This is a Scala example for image inference with a pre-trained caffe Inception_V1 model based
on Spark DataFrame (Dataset).

Analytics Zoo provides the DataFrame-based API for image reading, pre-processing.
The related classes followed the typical estimator/transformer pattern of Spark ML and can be used in
a standard Spark ML pipeline.

# Image Model Inference

You can run the full ModelInference example by following steps.

1. Prepare pre-trained model and definition file.

Download caffe inception v1 [weights](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
and [deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)
Use the path of the weights file as `$modelPath`, and the path of the prototxt as `$caffeDefPath`
when submitting spark jobs.

2. Prepare predict dataset

Put your image data for prediction in the ./predict folder. Alternatively, you may also use imagenet-2012
validation dataset to run the example, which can be found from <http://image-net.org/download-images>. After
you download the file (ILSVRC2012_img_val.tar), run the follow commands to extract the data.
```bash
    mkdir predict
    tar -xvf ILSVRC2012_img_val.tar -C ./predict/
```
3. Run this example

Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.example.nnframes.imageInference.ImageInferenceExample \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 32 \
    --folder ./predict
```

Command to run the example in Spark yarn mode:
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 8 \
    --num-executors 4 \
    --driver-memory 10g \
    --executor-memory 150g \
    --class com.intel.analytics.zoo.example.nnframes.ImageInference.ImageInference \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 32 \
    --folder hdfs://xxx
```

After inference, you should see something like this in the console:
```
+-----------------------------------------------------------------+----------+
|imageName                                                        |prediction|
+-----------------------------------------------------------------+----------+
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000001.JPEG|59.0      |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000002.JPEG|796.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000003.JPEG|231.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000005.JPEG|432.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000007.JPEG|335.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000009.JPEG|675.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000011.JPEG|110.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000012.JPEG|287.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000013.JPEG|371.0     |
|hdfs://Almaren-Node-069:9000/predict/ILSVRC2012_val_00000014.JPEG|758.0     |
+-----------------------------------------------------------------+----------+
```

To map the class to human readable text, please refer to https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt