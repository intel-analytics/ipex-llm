# Real-time image classification streaming using Analytics Zoo and Flink

- [Getting started this tutorial](#Getting-started-this-tutorial)  
  - [Summary](#Summary)
  - [Prerequisites](#Prerequisites)
- [Preparing model and data](#Preparing-model-and-data)  
  - [Obtaining model](#Obtaining-model)
  - [Loading and preprocessing images](#Loading-and-preprocessing-images)
- [Starting the image classification program on Flink ](#Starting-the-image-classification-program-on-Flink)  
  - [Obtaining an execution environment](#Obtaining-an-execution-environment)  
  - [Creating DataStream](#creating-datastream)  
  - [Executing transformation functions](#Executing-transformation-functions)  
    - [Defining an Analytics Zoo Inference Model](#Defining-an-Analytics-Zoo-Inference-Model)
    - [Specifying MapFunction](#Specifying-MapFunction)
    - [DataStream map transformation](#DataStream-map-transformation)
  - [Writing final results](#writing-final-results)  
  - [Triggering the program execution](#triggering-the-program-execution)    
- [Running the Flink program on a local machine or a cluster](#running-the-flink-program-on-a-local-machine-or-a-cluster)
  - [Configuring and starting Flink](#Configuring-and-starting-Flink)
  - [Building the project](#Building-the-project)
  - [Running the example](#Running-the-example)

## Getting started this tutorial

### Summary

This is the real-time image classification on Apache Flink streaming. Images extracted from ImageNet database will be predicted with pre-trained MobileNet_v1 model which is loaded by Analytics Zoo Inference Model.

ImageNet is a large-scale database designed for use in visual object detection research. It has more than 14 million images among more than 20,000 categories, such as keyboard, mouse, pencil, and animals.

MobileNet is efficient convolutional neural networks for mobile vision applications. It is based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. The mobilenet_v1 models are trained on ImageNet database. We select mobilenet_v1_1.0_224 which has an image input size of 224-by-224.

Analytics Zoo Inference Model package is aiming to provide high-level APIs to speed-up development.  It provides easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

Apache Flink is a powerful stream processing framework that supports batch processing, data streaming programs and running in common cluster environment. 

Let's get started this tutorial. We will use an example to introduce how to load pre-trained tensorflow model as Analytics Zoo Inference Model TFNet for the real-time image prediction on Flink streaming. See the whole program [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/ImageClassification). 

### Prerequisites

- **Environment preparation**

Make sure JDK 1.8, Flink 1.8.1, and Maven are installed. 

- **Building a project directory and installing Analytics Zoo** 

The example project contains: a POM file which is used by Maven to build the project; the source directory, which is `src/main/scala`.

Follow the [instructions](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) to install analytics-zoo for Scala project. Specify dependencies in the POM file. See the example [pom.xml](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/model-inference-examples/model-inference-flink/pom.xml) which is setting the required dependencies and configuration details of Analytics Zoo, Flink and scala for this project.

If you import the example project in the IDE(eg: IDEA), select **New - Project** from existing source, look through the example project directory and click OK, then select open as project in the window pop out next, using maven to build up the project.

## Preparing model and data

### Obtaining model

In the model repository of TensorFlow you can download multiple pre-trained weights of several different convolutional neural networks trained on ImageNet data. As mentioned above we are using a MobileNet in this tutorial. We can find them in the [MobileNet v1 description](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md#pre-trained-models) where we download [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz). Extract it with tar xf mobilenet_v1_1.0_224.tgz. In the folder you can see multiple files. 

```
├── mobilenet_v1_1.0_224.ckpt.data-00000-of-00001
├── mobilenet_v1_1.0_224.ckpt.index
├── mobilenet_v1_1.0_224.ckpt.meta
├── mobilenet_v1_1.0_224.tflite
├── mobilenet_v1_1.0_224_eval.pbtxt
├── mobilenet_v1_1.0_224_frozen.pb
└── mobilenet_v1_1.0_224_info.txt
```

We are only using mobilenet_v1_1.0_224_frozen.pb later. The details of input and output of the model can be viewed in mobilenet_v1_1.0_224_info.txt.

### Loading and preprocessing images

#### Knowing about dataset

The raw images in **ImageNet** are various sizes.  Let us show two of the predicting images.

<img src="../../../zoo/src/test/resources/imagenet/n02110063/n02110063_15462.JPEG" width="180">       
<img src="../../../zoo/src/test/resources/imagenet/n04370456/n04370456_5753.JPEG" width="180">

#### Preprocessing dataset

MobileNet_v1_1.0_224's input layer is 224 * 224 * 3. Inference Model requires predict input to be `JList[JList[JTensor]]`. 

In this example, `ImageProcessing` is prepared to provide approaches to convert format, resize and normalize. These methods are defined [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/ImageClassification/ImageProcessing.scala). 

First, let us load images from the image folder.

```scala
// load images from folder, and hold images as a list
val fileList = new File(imageDir).listFiles.toList
```

A  ImageProcessor class is created to apply methods from `trait ImageProcessing` to implement image pre-processing. View the class [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/model-inference-examples/model-inference-flink/src/main/scala/com/intel/analytics/zoo/apps/model/inference/flink/ImageClassification/ImageProcesser.scala).

```scala
class ImageProcessor extends ImageProcessing {
  def preProcess(bytes: Array[Byte], cropWidth: Int, cropHeight: Int ) = {
    // convert Array[byte] to OpenCVMat
    val imageMat = byteArrayToMat(bytes)
    // do a center crop by resizing a square
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    // convert OpenCVMat to Array
    val imageArray = matToArray(imageCent)
    // Normalize with channel and scale
    val imageNorm = channelScaledNormalize(imageArray, 127, 127, 127, 1/127f)
    imageNorm
  }
}
```

Each input image read from image folder is supposed to be converted as below, where each image read as Array[Byte]  and convert to `JList[JList[JTensor]]` .

```scala
// Image pre-processing
    val inputImages = fileList.map(file => {
      // Read image as Array[Byte]
      val imageBytes = FileUtils.readFileToByteArray(file)
      // Execute image processing with ImageProcessor class
      val imageProcess = new ImageProcessor
      val res = imageProcess.preProcess(imageBytes, 224, 224)
      // Convert input to List[List[JTensor]]]
      val input = new JTensor(res, Array(1, 224, 224, 3))
      List(util.Arrays.asList(input)).asJava
    })
```

## Starting the image classification program on Flink 

### Obtaining an execution environment

The first step is to create an execution environment. The `StreamExecutionEnvironment` is the context in which a streaming program is executed. `getExecutionEnvironment` is the typical function creating an environment to execute your program when the program is invoked on your local machine or a cluster.

```scala
val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
```

### Creating DataStream

`StreamExecutionEnvironment` provides several stream sources function. As we use `List` to hold the input images, we can create a DataStream from a collection using `fromCollection()` method.

```scala
// dataStream
val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputImages)
```

### Executing transformation functions

#### Defining an Analytics Zoo Inference Model

Analytics Zoo provides [Inference Model](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/inference.md) package for speeding up prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). You may see [here](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/APIGuide/PipelineAPI/inference.md) for more details of Inference Model APIs.

Define a class extended Analytics Zoo `InferenceModel`. We use the pre-trained frozen model of MobileNet and load it as TFNet in this example. Download the frozen model from [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz).  Unzip and extract mobilenet_v1_1.0_224_frozen.pb file.

The information about input and output nodes of the MobileNet can be found in the downloaded files at  ./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_info.txt

Let's define the input parameters of the class:

- **concurrentNum**-the number of requests a model can accept concurrently
- **modelPath**-the path of the pre-trained model
- **modelType**-the type of the model file format
- **inputs**-the input node(s) of the model
- **outputs**-the output node(s) of the model
- **intraOpParallelismThreads**- the num of intraOpParallelismThreads
- **interOpParallelismThreads**- the num of interOpParallelismThreads
- **usePerSessionThreads**- whether to perSessionThreads

```scala
// concurrentNum
var concurrentNum = 1

// modelPath 
var modelPath = "/path/to/model"

// modelType
var modelType = "frozenModel"

// inputs
var inputs = Array("input:0")

// outputs 
var outputs = Array("MobilenetV1/Predictions/Reshape_1:0")

// intraOpParallelismThreads
var intraOpParallelismThreads = 1

// interOpParallelismThreads
var interOpParallelismThreads = 1

// usePerSessionThreads
var usePerSessionThreads = true
```

Let's define a `MobileNetInferenceModel` class to extend Analytics Zoo `InferenceModel`.

```scala
class MobileNetInferenceModel(var concurrentNum: Int = 1, modelPath: String, modelType: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends InferenceModel(concurrentNum) with Serializable {
    
  // load the pre-trained tensorflow model as TFNet  
  doLoadTensorflow(modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
}
```

#### Specifying MapFunction

Define a class extends `RichMapFunction`. Three main methods of rich function in this example are open, close and map. `open()` is initialization method. `close()` is called after the last call to the main working methods. `map()` is the user-defined function, mapping an element from the input data set and to one exact element, ie, `JList[JList[JTensor]]`.

```scala
class ModelPredictionMapFunction(modelPath: String, modelType: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends RichMapFunction[JList[JList[JTensor]], Int] {
  var MobileNetInferenceModel: MobileNetInferenceModel = _

  // open
  override def open(parameters: Configuration): Unit = {
    MobileNetInferenceModel = new MobileNetInferenceModel(1, modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
  }

  // close
  override def close(): Unit = {
    MobileNetInferenceModel.doRelease()
  }

  // define map function with InferenceModel doPredict function
  // return predicted classes index
  override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = MobileNetInferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    (index)
  }
}
```

#### DataStream map transformation

Pass the `RichMapFunctionn`  to a `map` transformation.

```scala
val resultStream = dataStream.map(new ModelPredictionMapFunctionTFNet(savedModelBytes, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads))
```

### Writing final results

Obtain classification label by index. Print results to file or stdout.

```scala
// Obtain classfication label by index
    val results = resultStream.map(i => labels(i - 1))
// Print results to file or stdout
    if (params.has("output")) {
      results.writeAsText(params.get("output")).setParallelism(1)
    } else {
      println("Printing result to stdout. Use --output to specify output path.");
      results.print()
    }
```

### Triggering the program execution

The program is actually executed only when calling `execute()` on the `StreamExecutionEnvironment`. Whether the program is executed locally or submitted on a cluster depends on the type of execution environment.

```scala
env.execute()
```

At this step, we complete the whole program. Let's start how to run the example on a cluster.

## Running the Flink program on a local machine or a cluster

- ##### Configuring and starting Flink

Before start a flink cluster, edit /conf/flink-conf.yaml to set heap size or the number of task slots as you need, ie, `jobmanager.heap.size: 10g`, `taskmanager.numberOfTaskSlots: 2` 

You may start a flink cluster if there is no running one. Go to the location where you installed fink :

```scala
./bin/start-cluster.sh
```

Check the Dispatcher's web frontend at [http://localhost:8081](http://localhost:8081/) and make sure everything is up and running. To stop Flink when you're done type:

```scala
./bin/stop-cluster.sh
```

- ##### Building the project

Build the project using Maven. Go to the root directory of your inference flink project and execute the mvn clean package command, which prepares the jar file for your model inference flink program:

```scala
mvn clean package
```

The resulting jar file will be in the target subfolder like: target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar. 

- ##### Running the Example

Everything is ready. Let's run the following command with arguments to submit the Flink program. Change parameter settings as you need.

```shell
${FLINK_HOME}/bin/flink run \
    -m localhost:8081 -p 2 \
    -c com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification.ImageClassificationStreaming  \
    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --modelPath ${MODEL_PATH} --modelType "frozenModel"  \
    --images ${IMAGE_PATH} --classes ${CLASSES_FILE} --output ${OUTPUT}
```

View the result in your output file or stdout. The output should look similar as below if everything went according to plan.  It shows the prediction result.

```
Siberian husky
sweatshirt
```

#### Wrapping up

We have reached the end of the tutorial. In this tutorial, we introduce how to use Analytics Zoo Inference Model for image classification on streaming. We write a subclass extends `InferenceModel`, and load pre-trained model. With that, we define  `RichMapFunction` and started with the prediction on Flink streaming.

What goes for next? Try to take practice. Load the data and pre-trained model to see how the results get.
