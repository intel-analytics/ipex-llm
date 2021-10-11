Inference Model is a package in Analytics Zoo aiming to provide high-level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides Java, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

**Basic usage of Inference Model:**

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractInferenceModel` in Java).
2. Load pre-trained models with corresponding `load` methods, e.g, `doLoadBigDL` for Analytics Zoo, and `doLoadTensorflow` for TensorFlow.
3. Do prediction with `predict` method.

**OpenVINO requirements:**

[System requirements](https://software.intel.com/en-us/openvino-toolkit/documentation/system-requirements):

    Ubuntu 16.04.3 LTS or higher (64 bit)
    CentOS 7.6 or higher (64 bit)
    macOS 10.14 or higher (64 bit)

Python requirements:

    tensorflow>=1.2.0,<2.0.0
    networkx>=1.11
    numpy>=1.12.0
    defusedxml>=0.5.0
    test-generator>=0.1.1

**Supported models:**

1. [Analytics Zoo Models](https://analytics-zoo.github.io/master/##built-in-deep-learning-models)
2. [Caffe Models](https://github.com/BVLC/caffe/wiki/Model-Zoo)
3. [TensorFlow Models](https://github.com/tensorflow/models)
4. [OpenVINO models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

## **Load pre-trained model**
### **Load pre-trained Analytics Zoo model**
Load Analytics Zoo model with corresponding `load` methods (`load` for Java and Python, `doLoad` for Scala).

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadBigDL(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadBigDL(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_bigdl(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight. Default is `null`.

### **Load pre-trained Caffe model**
Load Caffe model with `loadCaffe` methods (`loadCaffe` for Java, `doLoadCaffe` for Scala and `load_caffe` Python).

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadCaffe(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadCaffe(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_caffe(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight.

### **Load pre-trained TensorFlow model**
Load model into `TFNet` with corresponding `loadTensorflow` methods (`loadTensorflow` for Java, `doLoadTensorflow` for Scala and `load_tensorflow` for Python)

We provide `loadTensorflow` with the following parameters:

* `modelPath`: String. Path of pre-trained model.
* `modelType`: String. Type of pre-trained model file.
* `Inputs`: Array[String]. The inputs of the model.
* `Outputs`: Array[String]. The outputs of the model.
* `intraOpParallelismThreads`: Int. The number of intraOpParallelismThreads.
* `interOpParallelismThreads`: Int. The number of interOpParallelismThreads.
* `usePerSessionThreads`: Boolean. Whether to perSessionThreads

Note that we prepare several implementations with less parameters based on this method, e.g., `loadTensorflow(modelPath, modelType)` for frozenModel.

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadTensorflow(modelPath, modelType);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadTensorflow(modelPath, modelType)
```

**Python**

```python
model = InferenceModel()
model.load_tensorflow(modelPath, modelType)
```

### **Load OpenVINO model**

Load OpenVINO model with `loadOpenVINO` methods (`loadOpenVINO` for Java, `doLoadOpenVINO` for Scala and `load_openvino` Python).

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadOpenVINO(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadOpenVINO(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_openvino(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained OpenVINO model.
* `weightPath`: String. Path of pre-trained OpenVINO model weight.

## **Predict with loaded model**
After loading pre-trained models with load methods, we can make prediction with unified `predict` method.

* `predictInput`: JList[JList[JTensor]] or [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) for Scale and Java, Numpy for Python. Input data for prediction. [JTensor](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java) is a 1D List, with Array[Int] shape.
* `predictOutput`: JList[JList[JTensor]] or [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) for Scale and Java, Numpy for Python. Prediction result.

Do prediction with `predict` methods (`predict` for Java and Python, `doPredict` for Scala).

**Java**

```java
List<List<JTensor>> predictOutput = model.predict(predictInput);
```

**Scala**

```scala
val predictOutput = model.doPredict(predictInput)
```

**Python**

```python
predict_output = model.predict(predict_input)
```
