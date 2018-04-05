
## Overview

NNFrames is a package in Analytics Zoo which is designed to provide DataFrame-based API in order to
facilitate Spark users and speed-up development. It extends DLFrame in BigDL and supports native integration with Spark ML Pipeline, which allows user to combine the
power of Analytics Zoo and Apache Spark MLlib for their application. NNFrames provides both Python and
Scala interfaces, and is compatible with both Spark 1.6 and Spark 2.x.


**Highlights**
1. Easy-to-use DataFrame(DataSet)-based API for training, prediction and evaluation with deep learning models.
2. Effortless integration with Spark ML pipeline and compatibility with other feature transformers and algorithms in Spark ML.
3. In a few lines, run large scale inference or transfer learning from pre-trained models of Caffe, Keras, Tensorflow or BigDL.
4. Training of customized model or BigDL built-in neural models (e.g. Inception, ResNet, Wide And Deep).
5. Rich toolset for feature extraction and processing, including image, audio and texts.

## Examples:

The examples are included in the Analytics Zoo source code.

1. image classification: model inference using pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/imageInference)

2. image classification: transfer learning from pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/transferLearning)

## Primary APIs

**NNEstimator**

Analytics Zoo provides `NNEstimator` for users with Apache Spark MLlib experience, which
provides high level API for training a BigDL Model with the Apache Spark
[Estimator](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#estimators)/
[Transfomer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers)
pattern, thus users can conveniently fit Analytics Zoo into a ML pipeline.

**NNClassifier and NNClassifierModel**
`NNClassifier` and `NNClassifierModel`extends `NNEstimator` and `DLModel` and focus on classification
tasks.

please check our
[NNEstimator API](NNEstimator.md) for detailed usage.

**NNImageReader and NNImageTransformer**
NNImageReader loads image into Spark DataFrame, and NNImageTransformer are used to perform the image pre-processing.

please check our
[ImageProcessing](ImagesProcessing.md) for detailed usage.
