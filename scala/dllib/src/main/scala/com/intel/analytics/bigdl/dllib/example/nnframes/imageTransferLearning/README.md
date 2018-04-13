# Overview
This is a Scala example for image transfer learning with a pre-trained caffe Inception_V1 model based
on Spark DataFrame (Dataset).

Analytics Zoo provides the DataFrame-based API for image reading, pre-processing.
The related classes followed the typical estimator/transformer pattern of Spark ML and can be used in
a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with Analytics Zoo.
For transfer learning, we will treat the inception-v1 model as a feature extractor and only
train a linear model on these features.

# Run the example

1. Prepare pre-trained model and defenition file.

Download caffe inception v1 [weights](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
and [deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)
Use the path of the weights file as `$modelPath`, and the path of the prototxt as `$caffeDefPath`
when submitting spark jobs.

2. Prepare dataset
Put your image data for training and validation in the ./data folder. For this example we
use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train dataset.
After you download the file (train.zip), run the follow commands to extract the data.

```bash
    unzip train.zip
```

2.3 Run this example

Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.example.nnframes.ImageTransferLearning.ImageTransferLearning \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 40 \
    --folder ./train \
    --nEpochs 20
```

Command to run the example in Spark yarn mode(TODO):
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 10 \
    --num-executors 4 \
    --driver-memory 10g \
    --executor-memory 150g \
    --class com.intel.analytics.zoo.example.nnframes.ImageTransferLearning.ImageTransferLearning \
    ./dist/lib/zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 40 \
    --folder hdfs://xxx \
    --nEpochs 20
```

```
+--------------------+-----+--------------------+--------------------+----------+
|               image|label|              output|           embedding|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[2.62691912666923...|       2.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[2.93451139441458...|       1.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[1.25100280001788...|       1.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[4.53599341199151...|       1.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[4.58738832094240...|       2.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[7.65169474448157...|       1.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[6.49528055873815...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[1.21466446216800...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[9.85538548547992...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[2.19180151361797...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[1.00327188192750...|       2.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[1.75065979419741...|       1.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[1.64268567459657...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[2.35099932410776...|       1.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[1.04405044112354...|       2.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[1.14560577912925...|       1.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[4.64494623884093...|       2.0|
|[hdfs://Almaren-N...|  2.0|[hdfs://Almaren-N...|[1.37401173105899...|       2.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[4.58144455706133...|       1.0|
|[hdfs://Almaren-N...|  1.0|[hdfs://Almaren-N...|[2.21465052163694...|       1.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 20 rows

evaluation result on validationDF: 0.9623
```

The model from transfer learning can achieve a 96.2% accuracy on the validation set.