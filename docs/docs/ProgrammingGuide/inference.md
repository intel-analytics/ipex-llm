## Inference Model

## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Tensorflow and Caffe.
Inference provides multiple Java/Scala interfaces.



## Highlights

1. Easy-to-use java/scala API for loading and prediction with deep learning models.

2. Support transformation of various input data type, thus supporting future prediction tasks

## Java Example

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

## Scala Example

```scala
import com.intel.analytics.zoo.pipeline.inference.FloatInferenceModel

class TextClassificationModel(var model: AbstractModule[Activity, Activity, Float],
                                @transient var predictor: LocalPredictor[Float]) extends FloatInferenceModel {
                                
}

TextClassificationModel model = new TextClassificationModel(model, predictor)
val result = model.predict(inputList)

```

