# Overview
This is a Scala example for image transfer learning with a pre-trained caffe Inception_V1 model based
on Spark DataFrame (Dataset).

BigDL provides the DataFrame-based API for image reading, pre-processing, model training and inference.
The related classes followed the typical estimator/transformer pattern of Spark ML and can be used in
a standard Spark ML pipeline. 

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with BigDL.
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
    mkdir data
    unzip -xvf train.tar -C ./data/
```

2.3 Run this example

Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.bigdl.example.DLFrames.imageTransferLearning.ImageTransferLearning \
    ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 32 \
    --folder ./data \
    --nEpochs 10
```

Command to run the example in Spark yarn mode(TODO):
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 8 \
    --num-executors 4 \
    --class com.intel.analytics.bigdl.example.DLFrames.imageTransferLearning.ImageTransferLearning \
    ./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
    --modelPath ./model/bvlc_googlenet.caffemodel \
    --caffeDefPath ./model/deploy.prototxt \
    --batchSize 32 \
    --folder ./data \
    --nEpochs 10
```