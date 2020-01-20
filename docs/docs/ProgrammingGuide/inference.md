Inference Model is a package in Analytics Zoo aiming to provide high-level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides Java, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

## **Load and predict with pre-trained model**
**Basic usage of Inference Model:**

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractInferenceModel` in Java).
2. Load pre-trained models with corresponding `load` methods, e.g, `doLoad` for Analytics Zoo, and `doLoadTF` for TensorFlow.
3. Do prediction with `predict` method.

**Supported models:**

1. [Analytics Zoo Models](https://analytics-zoo.github.io/master/##built-in-deep-learning-models)
2. [Caffe Models](https://github.com/BVLC/caffe/wiki/Model-Zoo)
3. [TensorFlow Models](https://github.com/tensorflow/models)
4. [OpenVINO models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

**Predict input and output**

* `predictInput`: JList[JList[JTensor]] or [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) for Scale and Java, Numpy for Python. Input data for prediction. [JTensor](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java) is a 1D List, with Array[Int] shape.
* `predictOutput`: JList[JList[JTensor]] or [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) for Scale and Java, Numpy for Python. Prediction result.


**OpenVINO requirements:**

[System requirements](https://software.intel.com/en-us/openvino-toolkit/documentation/system-requirements):

    Ubuntu 18.04 LTS (64 bit)
    CentOS 7.4 (64 bit)
    macOS 10.13, 10.14 (64 bit)

Python requirements:

    tensorflow>=1.2.0
    networkx>=1.11
    numpy>=1.12.0
    protobuf==3.6.1

**Java**

Write a subclass that extends `AbstractInferenceModel`, implement or override methods. Then, load model with corresponding `load` methods (load Analytics Zoo, caffe, OpenVINO and TensorFlow model with `load`, `loadCaffe`, `doLoadOpenVINO` and `loadTF`), and do prediction with `predict` method. 

```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class ExtendedInferenceModel extends AbstractInferenceModel {
    public ExtendedInferenceModel() {
        super();
    }
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
// Load Analytics Zoo model
model.load(modelPath, weightPath);
// Predict
List<List<JTensor>> result = model.predict(inputList);
```

**Scala**

New an instance of `InferenceModel`, and load model with corresponding `load` methods (load Analytics Zoo, caffe, OpenVINO and TensorFlow model with `doLoad`, `doLoadCaffe`, `doLoadOpenVINO` and `doLoadTF`), then do prediction with `predict` method.

```scala
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

val model = new InferenceModel()
// Load Analytics Zoo model
model.doLoad(modelPath, weightPath)
// Predict
val result = model.doPredict(inputList)
```

In some cases, you may want to write a subclass that extends `InferenceModel`, implement or override methods. Then, load model with corresponding `load` methods, and do prediction with `predict` method.

```scala
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class ExtendedInferenceModel extends InferenceModel {

}

val model = new ExtendedInferenceModel()
// Load Analytics Zoo model
model.doLoad(modelPath, weightPath)
// Predict
val result = model.doPredict(inputList)
```

**Python**

New an instance of `InferenceModel`, and load Zoo model with corresponding `load` methods (load Analytics Zoo, caffe, OpenVINO and TensorFlow model with `load`, `load_caffe`, `load_openvino` and `load_tf`), then do prediction with `predict` method.

```python
from zoo.pipeline.inference import InferenceModel

model = InferenceModel()
# Load Analytics Zoo model
model.load(model_path, weight_path)
# Predict
result = model.predict(input_list)
```

In some cases, you may want to write a subclass that extends `InferenceModel`, implement or override methods. Then, load model with corresponding `load` methods, and do prediction with `predict` method.

```python
from zoo.pipeline.inference import InferenceModel

class ExtendedInferenceModel(InferenceModel):

    def __init__(self):
        pass

model = ExtendedInferenceModel()
# Load Analytics Zoo model
model.load(model_path, weight_path)
# Predict
result = model.predict(input_list)
```

## **Examples**
We provide examples based on InferenceModel.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples) for the Java example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/streaming/textclassification) for the Scala example.
