
## Overview

DLFrames is a package in BigDL which is designed to provide DataFrame-based API in order to
facilitate Spark users and speed-up development. It extends the common infrastructure in Apache
Spark and supports native integration with Spark ML Pipeline, which allows user to combine the
power of BigDL and Apache Spark MLlib for their application. DLFrames provides both Python and
Scala interfaces, and is compatible with both Spark 1.6 and Spark 2.x.


**Highlights**
1. Easy-to-use DataFrame(DataSet)-based API for training, prediction and evaluation with deep learning models.
2. Effortless integration with Spark ML pipeline and compatibility with other feature transformers and algorithms in Spark ML.
3. In a few lines, run large scale inference or transfer learning from pre-trained models of Caffe, Keras, Tensorflow or BigDL.
4. Training of customized model or BigDL built-in neural models (e.g. Inception, ResNet, Wide And Deep).
5. Rich toolset for feature extraction and processing, including image, audio and texts.

## Examples:

The examples are included in the BigDL source code, please adjust the branch on Github according to your BigDL version.

1. image classification: model inference using pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/dlframes/imageInference)
    [Python version](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/examples/dlframes/ImageInference)

2. image classification: transfer learning from pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/dlframes/transferLearning)
    [Python version](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/examples/dlframes/transferLearning)

3. Use BigDL to train a simple linear model:
    [Scala version](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/dlframes)
    

## Primary APIs

**DLEstimator and DLModel**

BigDL provides `DLEstimator` and `DLModel` for users with Apache Spark MLlib experience, which
provides high level API for training a BigDL Model with the Apache Spark
[Estimator](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#estimators)/
[Transfomer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers)
pattern, thus users can conveniently fit BigDL into a ML pipeline. The fitted model `DLModel`
contains the trained BigDL model and extends the Spark ML `Model` class.
Alternatively users may also construct a `DLModel` with a pre-trained BigDL model to use it in
Spark ML Pipeline for prediction. 

**DLClassifier and DLClassifierModel**
`DLClassifier` and `DLClassifierModel`extends `DLEstimator` and `DLModel` and focus on classification
tasks. 

please check our
[ML Pipeline API](../APIGuide/DLFrames/DLEstimator_DLClassifier.md) for detailed usage.

**DLImageReader and DLImageTransformer**
DLImageReader loads image into Spark DataFrame, and DLImageTransformer are used to perform the image pre-processing.

please check our
[ImageProcessing](../APIGuide/DLFrames/ImagesProcessing.md) for detailed usage.

