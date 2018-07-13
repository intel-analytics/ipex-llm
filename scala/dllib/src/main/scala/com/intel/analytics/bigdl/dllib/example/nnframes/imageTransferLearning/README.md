# Overview
This is a Scala example for image transfer learning with a pre-trained caffe Inception_V1 model
based on Spark DataFrame (Dataset).

Analytics Zoo provides the DataFrame-based API for image reading, preprocessing, model training
and inference. The related classes followed the typical estimator/transformer pattern of Spark
ML and can be used in a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with
Analytics Zoo. For transfer learning, we will treat the inception-v1 model as a feature extractor
and only train a linear model on these features.

# Run the example

1. Prepare pre-trained model and defenition file.

Download caffe inception v1 [weights](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
and [deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)
Use the path of the weights file as `$modelPath`, and the path of the prototxt as `$caffeDefPath`
when submitting spark jobs.

2. Prepare dataset
Put your image data for training and validation in the ./data folder. For this example we
use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train dataset.
The following commands copy about 1100 images of cats and dogs into demo/cats and demo/dogs
separately.

```bash
unzip train.zip
mkdir -p demo/dogs
mkdir -p demo/cats
cp train/cat.7* demo/cats
cp train/dog.7* demo/dogs
```

2.3 Run this example

Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[2] \
    --driver-memory 2g \
    --class com.intel.analytics.zoo.examples.nnframes.imageTransferLearning.ImageTransferLearning \
    ./dist/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 40 \
    --folder ./demo \
    --nEpochs 20
```

Command to run the example in Spark yarn mode(TODO):
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 1 \
    --num-executors 4 \
    --driver-memory 2g \
    --executor-memory 2g \
    --class com.intel.analytics.zoo.examples.nnframes.imageTransferLearning.ImageTransferLearning \
    ./dist/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 40 \
    --folder hdfs://xxx \
    --nEpochs 20
```

```
+--------------------+-----+--------------------+----------+
|               image|label|           embedding|prediction|
+--------------------+-----+--------------------+----------+
|[file:/some/path...|  1.0|[8.865501E-6, 1.3...|       1.0|
|[file:/some/path...|  1.0|[5.498128E-6, 1.8...|       1.0|
|[file:/some/path...|  1.0|[1.594756E-5, 4.1...|       1.0|
|[file:/some/path...|  1.0|[1.6185285E-6, 3....|       1.0|
|[file:/some/path...|  1.0|[2.6421341E-5, 7....|       1.0|
|[file:/some/path...|  1.0|[1.9225668E-6, 4....|       1.0|
|[file:/some/path...|  1.0|[2.7568094E-5, 1....|       1.0|
|[file:/some/path...|  1.0|[3.545212E-4, 1.4...|       1.0|
|[file:/some/path...|  1.0|[3.261492E-5, 3.8...|       1.0|
|[file:/some/path...|  1.0|[2.6861264E-6, 1....|       1.0|
|[file:/some/path...|  1.0|[4.7644085E-6, 8....|       1.0|
|[file:/some/path...|  1.0|[2.7128006E-5, 1....|       2.0|
|[file:/some/path...|  1.0|[4.6807084E-8, 7....|       1.0|
|[file:/some/path...|  1.0|[2.6069986E-6, 6....|       1.0|
|[file:/some/path...|  1.0|[2.586313E-7, 1.6...|       1.0|
|[file:/some/path...|  1.0|[5.149611E-6, 1.1...|       1.0|
|[file:/some/path...|  1.0|[1.3761636E-5, 6....|       1.0|
|[file:/some/path...|  1.0|[1.2598471E-6, 6....|       2.0|
|[file:/some/path...|  1.0|[2.3851356E-7, 3....|       1.0|
|[file:/some/path...|  1.0|[1.426664E-6, 2.1...|       1.0|
+--------------------+-----+--------------------+----------+

only showing top 20 rows

evaluation result on validationDF: 0.9623
```

The model from transfer learning can achieve a 96.2% accuracy on the validation set.
