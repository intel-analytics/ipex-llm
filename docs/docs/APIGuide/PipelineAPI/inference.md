## Inference Model

## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Tensorflow and Caffe.
Inference provides multiple Java/Scala interfaces.



## Highlights

1. Easy-to-use java/scala API for loading and prediction with deep learning models.

2. Support transformation of various input data type, thus supporting future prediction tasks

## Primary APIs for Java


**load**

AbstractInferenceModel provides `load` API for loading a pre-trained model,
thus we can conveniently load various kinds of pre-trained models in java applications. The load result of
`AbstractInferenceModel` is a [`FloatInferenceModel`](https://github.com/xuex2017/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/scala/com/intel/analytics/zoo/pipeline/inference/FloatInferenceModel.scala). 
We just need to specify the model path and optionally weight path if exists where we previously saved the model.


**predict**

AbstractInferenceModel provides `predict` API for prediction with loaded model.
The predict result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.


**Java example**

It's very easy to apply abstract inference model for inference with below code piece.
You will need to write a subclass that extends AbstractinferenceModel.
```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
 }
TextClassificationModel model = new TextClassificationModel();
model.load(modelPath, weightPath);
List<List<JTensor>> result = model.predict(inputList);
```

## Primary APIs for Scala

**InferenceSupportive**

`InferenceSupportive` is a trait containing several methods for type transformation, which transfer a model input 
to a valid data type, thus supporting future inference model prediction tasks.

For example, method `transferTensorToJTensor` convert a model input of data type `Tensor` 
to [`JTensor`](https://github.com/intel-analytics/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java)
, which will be the input for a FloatInferenceModel.

**FloatInferenceModel**

`FloatInferenceModel` is an extending class of `InferenceSupportive` and additionally provides `predict` API for prediction tasks.

**InferenceModelFactory**

`InferenceModelFactory` is an object with APIs for loading pre-trained Analytics Zoo models, Caffe models and Tensorflow models.
We just need to specify the model path and optionally weight path if exists where we previously saved the model.
The load result of is a `FloatInferenceModel`.


**ModelLoader**

`ModelLoader` is an extending object of  `InferenceSupportive` and focus on the implementation of loading pre-trained models