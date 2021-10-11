

## Overview

NNFrames is a package in Analytics Zoo aiming to provide DataFrame-based high level API to
facilitate Spark users and speed-up development. It supports native integration with Spark ML
Pipeline, which allows user to combine the power of Analytics Zoo, BigDL and Apache Spark MLlib.
NNFrames provides both Python and Scala interfaces, and is compatible with both Spark 1.6 and
Spark 2.x.


**Highlights**

1. Easy-to-use DataFrame(DataSet)-based API for training, prediction and evaluation with deep learning models.

2. Effortless integration with Spark ML pipeline and compatibility with other feature transformers and algorithms in Spark ML.

3. In a few lines, run large scale inference or transfer learning from pre-trained models of Caffe, Keras, Tensorflow or BigDL.

4. Training of customized model or BigDL built-in neural models (e.g. Inception, ResNet, Wide And Deep).

5. Rich toolset for feature extraction and processing, including image, audio and texts.


## Examples:

The examples are included in the Analytics Zoo source code.

1. image classification: model inference using pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/imageInference)
    [Python version](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/nnframes/imageInference)
2. image classification: transfer learning from pre-trained Inception v1 model.
    [Scala version](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/imageTransferLearning)
    [Python version](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/nnframes/imageTransferLearning)

## Primary APIs

**NNEstimator and NNModel**

Analytics Zoo provides `NNEstimator` for model training with Spark DataFrame, which
provides high level API for training a BigDL Model with the Apache Spark
[Estimator](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#estimators)/
[Transfomer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers)
pattern, thus users can conveniently fit Analytics Zoo into a ML pipeline. The fit result of
`NNEstimator` is a NNModel, which is a Spark ML Transformer.

please check our
[NNEstimator API](../APIGuide/PipelineAPI/nnframes.md) for detailed usage.

**NNClassifier and NNClassifierModel**
`NNClassifier` and `NNClassifierModel`extends `NNEstimator` and `NNModel` and focus on 
classification tasks, where both label column and prediction column are of Double type.

**NNImageReader**
NNImageReader loads image into Spark DataFrame.

please check our
[ImageProcessing](../APIGuide/PipelineAPI/nnframes.md#NNImageReader) for detailed usage.
