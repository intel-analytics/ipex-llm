# Inference Model
Inference Model is a package in Analytics Zoo aiming to provide high-level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides Java, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

**Basic usage of Inference Model:**

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractInferenceModel` in Java).
2. Load pre-trained models with corresponding `load` methods, e.g, doLoad for Analytics Zoo, and doLoadTF for TensorFlow.
3. Do prediction with `predict` method.

**OpenVINO requirements:**

    tensorflow>=1.2.0
    mxnet>=1.0.0,<=1.3.1
    networkx>=1.11
    numpy>=1.12.0
    protobuf==3.6.1
    onnx>=1.1.2

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
model.load(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoad(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load(modelPath, weightPath)
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

### **Load TensorFlow model**
There are two backends to load a tensorflow model: TensorFlow and OpenVINO. When using TensorFlow as backend, tensorflow model will be loaded into `TFNet`. Otherwise, it will be coverted into OpenVINO model, and loaded into `OpenVINOModel`.

**1. Load with TensorFlow backend**

Load model into `TFNet` with TensorFlow backend, with corresponding `loadTF` methods (`loadTF` for Java, `doLoadTF` for Scala and `load_tf` Python)

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadTF(modelPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadTF(modelPath)
```

**Python**

```python
model = InferenceModel()
model.load_tf(modelPath)
```

* `modelPath`: String. Path of pre-trained model.

**2. Load with OpenVINO backend**

Load model into `OpenVINOModel` with OpenVINO backend, with corresponding `loadTF` methods (`loadTF` for Java, `doLoadTF` for Scala and `load_tf` Python). Note that OpenVINO cannot directly load TensorFlow models. We need to [covert TensorFlow models into OpenVINO models]((https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)), then load models into OpenVINO.

Herein Analytics Zoo, we merge these two steps into one, and provide `loadOpenVINOModelForTF` with the following parameters:

* `modelPath`: String. Path of pre-trained tensorflow model.
* `modelType`: String. Type the type of the tensorflow model.
* `checkpointPath`: String. Path of the tensorflow checkpoint file
* `inputShape`: Array[Int]. Input shape that should be fed to an input node(s) of the model
* `ifReverseInputChannels`: Boolean. If need reverse input channels. switch the input channels order from RGB to BGR (or vice versa).
* `meanValues`: Array[Int]. All input values coming from original network inputs will be divided by this value.
* `scale`: Float. Scale value, to be used for the input image per channel.
* `outputDir`: String. Path of pre-trained tensorflow model.

Note that we prepare several implementations with less parameters based on this method, e.g., `loadTF(modelPath, modelType)`.

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadTF(modelPath, modelType);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadTF(modelPath, modelType)
```

**Python**

```python
model = InferenceModel()
model.load_tf(modelPath, modelType)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight.

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

### **predict**

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

### **loadTFAsCalibratedOpenVINO**

Do prediction with int8 optimized model. Powered by [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) and [Intel Deep Learning Boost](https://www.intel.ai/intel-deep-learning-boost/). Currently, this API is only for OpenVINO. For Analytics Zoo model, int8 optimized model can directly make prediction with `predict` method.

To load an OpenVINO int8 optimized model from TensorFlow, we build `loadTFAsCalibratedOpenVINO` methods with 4 more parameters than `loadOpenVINOModelForTF`.

* `networkType`: String. Type of an inferred network, "C" to calibrate Classification, "OD" to calibrate Object Detection, "RawC" to collect only statistics for Classification, "RawOD" to collect only statistics for Object Detection.
* `validationFilePath`: String. Path to a file with validation images path and target labels.
* `subset`: Int. Number of pictures from the whole validation set to create the calibration dataset.
* `opencvLibPath`: String. lib path where libopencv_imgcodecs.so.4.0, libopencv_core.so.4.0 and libopencv_imgproc.so.4.0 can be found.

## **Supportive classes**

**InferenceModel**

`doOptimizeTF` method in Scala is designed for coverting TensorFlow model into OpenVINO model.

`doCalibrateTF` method in Scala is designed for optimizing OpenVINO model into OpenVINO int8 optimized model.

Pipline of these API:

TensorFlow model -`doOptimizeTF`-> OpenVINO model -`doCalibrateTF`-> OpenVINO int8 optimized model

**InferenceSupportive**

`InferenceSupportive` is a trait containing several methods for type transformation, which transfer a model input 
to a valid data type, thus supporting future inference model prediction tasks.

For example, method `transferTensorToJTensor` convert a model input of data type `Tensor` 
to [`JTensor`](https://github.com/intel-analytics/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java)
, which will be the input for a FloatInferenceModel.

**AbstractModel**

`AbstractModel` is an abstract class to provide APIs for basic functions - `predict` interface for prediction, `copy` interface for coping the model into the queue of AbstractModels, `release` interface for releasing the model and `isReleased` interface for checking the state of model release.  

**FloatModel**

`FloatModel` is an extending class of `AbstractModel` and achieves all `AbstractModel` interfaces. 

**OpenVINOModel**

`OpenVINOModel` is an extending class of `AbstractModel`. It achieves all `AbstractModel` functions.

**InferenceModelFactory**

`InferenceModelFactory` is an object with APIs for loading pre-trained Analytics Zoo models, Caffe models, Tensorflow models and OpenVINO Intermediate Representations(IR). Analytics Zoo models, Caffe models, Tensorflow models can be loaded as FloatModels. The load result of it is a `FloatModel` Tensorflow models and OpenVINO Intermediate Representations(IR) can be loaded as OpenVINOModels. The load result of it is an `OpenVINOModel`. 
The load result of it is a `FloatModel` or an `OpenVINOModel`.

**OpenVinoInferenceSupportive**

`OpenVinoInferenceSupportive` is an extending object of `InferenceSupportive` and focuses on the implementation of loading pre-trained models, including tensorflow models and OpenVINO Intermediate Representations(IR).
