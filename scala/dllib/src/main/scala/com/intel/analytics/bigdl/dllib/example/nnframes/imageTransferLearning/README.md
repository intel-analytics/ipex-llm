## Overview
This is a Scala example for image transfer learning with a pre-trained caffe Inception_V1 model
based on Spark DataFrame (Dataset).

Analytics Zoo provides the DataFrame-based API for image reading, preprocessing, model training
and inference. The related classes followed the typical estimator/transformer pattern of Spark
ML and can be used in a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with
Analytics Zoo. For transfer learning, we will treat the inception-v1 model as a feature extractor
and only train a linear model on these features.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Run the example

1. Prepare pre-trained model and defenition file.
Download caffe inception v1 [weights](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
and [deploy.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)
Put the files in `/tmp/zoo` or other folder.

2. Prepare dataset
For this example we use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train
dataset. Download the data and run the following commands to copy about 1100 images of cats
and dogs into `samples` folder.

```bash
unzip train.zip -d /tmp/zoo/dogs_cats
cd /tmp/zoo/dogs_cats
mkdir samples
cp train/cat.7* samples
cp train/dog.7* samples
```
`7` is randomly chosen and can be replaced with other digit.

3. Run this example

Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode, adjust
 the memory size according to your image:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[2] \
    --driver-memory 5g \
    --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageTransferLearning.ImageTransferLearning \
    --caffeDefPath /tmp/zoo/deploy.prototxt \
    --caffeWeightsPath /tmp/zoo/bvlc_googlenet.caffemodel \
    --batchSize 32 \
    --imagePath /tmp/zoo/dogs_cats/samples \
    --nEpochs 20
```

After training, you should see something like this in the console:

```
+--------------------+-----+--------------------+----------+
|               image|label|           embedding|prediction|
+--------------------+-----+--------------------+----------+
|[file:/tmp/zoo/do...|  1.0|[8.865501E-6, 1.3...|       1.0|
|[file:/tmp/zoo/do...|  1.0|[5.498128E-6, 1.8...|       1.0|
|[file:/tmp/zoo/do...|  1.0|[6.2580126E-5, 7....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[2.2474323E-7, 3....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[9.706464E-7, 1.2...|       1.0|
|[file:/tmp/zoo/do...|  1.0|[1.8358279E-6, 7....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[1.8146375E-4, 2....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[2.746296E-5, 3.0...|       1.0|
|[file:/tmp/zoo/do...|  1.0|[1.5992642E-5, 3....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[5.7681555E-6, 1....|       1.0|
|[file:/tmp/zoo/do...|  1.0|[2.423017E-6, 3.9...|       1.0|
|[file:/tmp/zoo/do...|  1.0|[3.6148306E-6, 5....|       1.0|
|[file:/tmp/zoo/do...|  2.0|[8.580417E-5, 1.9...|       2.0|
|[file:/tmp/zoo/do...|  2.0|[3.536114E-6, 1.4...|       2.0|
|[file:/tmp/zoo/do...|  2.0|[4.0955474E-5, 2....|       2.0|
|[file:/tmp/zoo/do...|  2.0|[2.6370296E-6, 7....|       2.0|
|[file:/tmp/zoo/do...|  2.0|[4.9449877E-8, 1....|       2.0|
|[file:/tmp/zoo/do...|  2.0|[1.4347982E-6, 1....|       2.0|
+--------------------+-----+--------------------+----------+

evaluation result on validationDF: 0.9833821330950844
```

The model from transfer learning can achieve high accuracy on the validation set.

In this example, we use the Inception-V1 model. Please feel free to explore other models from
Caffe, Keras and Tensorflow. Analytics Zoo provides popular pre-trained model in https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/#download-link.
