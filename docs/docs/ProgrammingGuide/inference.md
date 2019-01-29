## Inference Model

## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
Inference provides multiple Scala interfaces.


## Highlights

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

2. Support transformation of various input data type, thus supporting future prediction tasks.

3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

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
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class TextClassificationModel extends InferenceModel {

}

val model = new TextClassificationModel()
model.doLoad(modelPath, weightPath)
val result = model.doPredict(inputList)

```

