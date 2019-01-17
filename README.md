<div align="center">
   <p align="center"> <img src="https://github.com/analytics-zoo/analytics-zoo.github.io/blob/master/img/logo.jpg" height=240px； weight=320px；"><br></p>
</div>
      
_A unified analytics + AI platform for **distributed TensorFlow, Keras and BigDL on Apache Spark**_



---

## What is Analytics Zoo?
__Analytics Zoo__ provides a unified analytics + AI platform that seamlessly unites *__Spark, TensorFlow, Keras and BigDL__* programs into an integrated pipeline; the entire pipeline can then transparently scale out to a large Hadoop/Spark cluster for distributed training or inference. 
- _Data wrangling and analysis using PySpark_
- _Deep learning model development using TensorFlow or Keras_
- _Distributed training/inference on Spark and BigDL_
- _All within a single unified pipeline and in a user-transparent fashion!_

In addition, Analytics Zoo also provides a rich set of analytics and AI support for the end-to-end pipeline, including:
- *Easy-to-use abstractions and APIs* (e.g., transfer learning support, autograd operations, Spark DataFrame and ML pipeline support, online model serving API, etc.) 
- *Common feature engineering operations* (for image, text, 3D image, etc.)
- *Built-in deep learning models* (e.g., object detection, image classification, text classification, recommendation, anomaly detection, text matching, etc.) 
- *Reference use cases* (e.g., anomaly detection, sentiment analysis, fraud detection, image similarity, etc.)

## How to use Analytics Zoo?
- To get started, please refer to the [Python install guide](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) or [Scala install guide](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/).

- For running distributed TensorFlow/Keras on Spark and BigDL, please refer to the quick start [here](#distributed-tensorflow-and-keras-on-sparkbigdl) and the details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/tensorflow/).

- For more information, You may refer to the [Analytics Zoo document website](https://analytics-zoo.github.io/master/).

- For additional questions and discussions, you can join the [Google User Group](https://groups.google.com/forum/#!forum/bigdl-user-group) (or subscribe to the [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)).

---

## Overview
- [Distributed Tensorflow and Keras on Spark/BigDL](#distributed-tensorflow-and-keras-on-sparkbigdl)
  - Data wrangling and analysis using PySpark
  - Deep learning model development using TensorFlow or Keras
  - Distributed training/inference on Spark and BigDL
  - All within a single unified pipeline and in a user-transparent fashion!

- [High level abstractions and APIs](#high-level-abstractions-and-apis)
  - [Transfer learning](#transfer-learning): customize pretained model for *feature extraction or fine-tuning*
  - [`autograd`](#autograd): build custom layer/loss using *auto differentiation operations* 
  - [`nnframes`](#nnframes): native deep learning support in *Spark DataFrames and ML Pipelines*
  - [Model serving](#model-serving): productionize *model serving and inference* using [POJO](https://en.wikipedia.org/wiki/Plain_old_Java_object) APIs
  
- [Built-in deep learning models](#built-in-deep-learning-models)
  - [Object detection API](#object-detection-api): high-level API and pretrained models (e.g., SSD and Faster-RCNN) for *object detection*
  - [Image classification API](#image-classification-api): high-level API and pretrained models (e.g., VGG, Inception, ResNet, MobileNet, etc.) for *image classification*
  - [Text classification API](#text-classification-api): high-level API and pre-defined models (using CNN, LSTM, etc.) for *text classification*
  - [Recommedation API](#recommendation-api): high-level API and pre-defined models (e.g., Neural Collaborative Filtering, Wide and Deep Learning, etc.) for *recommendation*
  - [Anomaly detection API](#anomaly-detection-api): high-level API and pre-defined models based on LSTM for *anomaly detection*
  - [Text matching API](#text-matching-api): high-level API and pre-defined KNRM model for *text matching*
  
- [Reference use cases](#reference-use-cases): a collection of end-to-end *reference use cases* (e.g., anomaly detection, sentiment analysis, fraud detection, image augmentation, object detection, variational autoencoder, etc.)

## _Distributed Tensorflow and Keras on Spark/BigDL_
To make it easy to build and productionize the deep learning applications for Big Data, Analytics Zoo provides a unified analytics + AI platform that seamlessly unites Spark, TensorFlow, Keras and BigDL programs into an integrated pipeline (as illustrated below), which can then transparently run on a large-scale Hadoop/Spark clusters for distributed training and inference. (Please see more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/tensorflow/)).

1. Data wrangling and analysis using PySpark
   ```python
   from zoo import init_nncontext
   from zoo.pipeline.api.net import TFDataset

   sc = init_nncontext()

   #Each record in the train_rdd consists of a list of NumPy ndrrays
   train_rdd = sc.parallelize(file_list)
     .map(lambda x: read_image_and_label(x))
     .map(lambda image_label: decode_to_ndarrays(image_label))

   #TFDataset represents a distributed set of elements,
   #in which each element contains one or more Tensorflow Tensor objects. 
   dataset = TFDataset.from_rdd(train_rdd,
                                names=["features", "labels"],
                                shapes=[[28, 28, 1], [1]],
                                types=[tf.float32, tf.int32],
                                batch_size=BATCH_SIZE)
   ```

2. Deep learning model development using TensorFlow

   ```python
   import tensorflow as tf

   slim = tf.contrib.slim

   images, labels = dataset.tensors
   labels = tf.squeeze(labels)
   with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

   loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
   ```

3. Distributed training on Spark and BigDL
   ```python
   from zoo.pipeline.api.net import TFOptimizer
   from bigdl.optim.optimizer import MaxIteration, Adam, MaxEpoch, TrainSummary

   optimizer = TFOptimizer(loss, Adam(1e-3))
   optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
   optimizer.optimize(end_trigger=MaxEpoch(5))
   ```

4. Alternatively, using Keras APIs for model development and distributed training
   ```python
   from zoo.pipeline.api.keras.models import *
   from zoo.pipeline.api.keras.layers import *

   model = Sequential()
   model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
   model.add(Convolution2D(6, 5, 5, activation="tanh", name="conv1_5x5"))
   model.add(MaxPooling2D())
   model.add(Convolution2D(12, 5, 5, activation="tanh", name="conv2_5x5"))
   model.add(MaxPooling2D())
   model.add(Flatten())
   model.add(Dense(100, activation="tanh", name="fc1"))
   model.add(Dense(class_num, activation="softmax", name="fc2"))

   model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam')
   model.fit(train_rdd, batch_size=BATCH_SIZE, nb_epoch=5)
   ```

## _High level abstractions and APIs_
Analytics Zoo provides a set of easy-to-use, high level abstractions and APIs that natively transfer learning, autograd and custom layer/loss, Spark DataFrames and ML Pipelines, online model serving, etc. etc.

### _Transfer learning_
Using the high level transfer learning APIs, you can easily customize pretrained models for *feature extraction or fine-tuning*. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/transferlearning/))

1. Load an existing model (pretrained in Caffe)
   ```python
   from zoo.pipeline.api.net import *
   full_model = Net.load_caffe(def_path, model_path)
   ```

2. Remove the last few layers
   ```python
   # create a new model by removing layers after pool5/drop_7x7_s1
   model = full_model.new_graph(["pool5/drop_7x7_s1"])
   ```

3. Freeze the first few layers
   ```python
   # freeze layers from input to pool4/3x3_s2 inclusive
   model.freeze_up_to(["pool4/3x3_s2"])
   ```

4. Add a few new layers
   ```python
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   inputs = Input(name="input", shape=(3, 224, 224))
   inception = model.to_keras()(inputs)
   flatten = Flatten()(inception)
   logits = Dense(2)(flatten)
   newModel = Model(inputs, logits)
   ```

### _`autograd`_
`autograd` provides automatic differentiation for math operations, so that you can easily build your own *custom loss and layer* (in both Python and Scala), as illustracted below. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/autograd/))

1. Define model using Keras-style API and `autograd` 
   ```python
   import zoo.pipeline.api.autograd as A
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *

   input = Input(shape=[2, 20])
   features = TimeDistributed(layer=Dense(30))(input)
   f1 = features.index_select(1, 0)
   f2 = features.index_select(1, 1)
   diff = A.abs(f1 - f2)
   model = Model(input, diff)
   ```

2. Optionally define custom loss function using `autograd`
   ```python
   def mean_absolute_error(y_true, y_pred):
       return mean(abs(y_true - y_pred), axis=1)
   ```

3. Train model with *custom loss function*
   ```python
   model.compile(optimizer=SGD(), loss=mean_absolute_error)
   model.fit(x=..., y=...)
   ```

### _`nnframes`_
`nnframes` provides *native deep learning support in Spark DataFrames and ML Pipelines*, so that you can easily build complex deep learning pipelines in just a few lines, as illustrated below. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/nnframes/))

1. Initialize *NNContext* and load images into *DataFrames* using `NNImageReader`
   ```python
   from zoo.common.nncontext import *
   from zoo.pipeline.nnframes import *
   from zoo.feature.image import *
   sc = init_nncontext()
   imageDF = NNImageReader.readImages(image_path, sc)
   ```

2. Process loaded data using *DataFrames transformations*
   ```python
   getName = udf(lambda row: ...)
   getLabel = udf(lambda name: ...)
   df = imageDF.withColumn("name", getName(col("image"))).withColumn("label", getLabel(col('name')))
   ```

3. Processing image using built-in *feature engineering operations*
   ```
   transformer = RowToImageFeature() -> ImageResize(64, 64) -> ImageChannelNormalize(123.0, 117.0, 104.0) \
                 -> ImageMatToTensor() -> ImageFeatureToTensor())
   ```

4. Define model using *Keras-style APIs*
   ```python
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   model = Sequential().add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28))) \
                   .add(MaxPooling2D(pool_size=(2, 2))).add(Flatten()).add(Dense(10, activation='softmax')))
   ```

5. Train model using *Spark ML Pipelines*
   ```python
   classifier = NNClassifier(model, CrossEntropyCriterion(),transformer).setLearningRate(0.003) \
                   .setBatchSize(40).setMaxEpoch(1).setFeaturesCol("image").setCachingSample(False)
   nnModel = classifier.fit(df)
   ```
   

### _Model Serving_
Using the [POJO](https://en.wikipedia.org/wiki/Plain_old_Java_object) model serving API, you can productionize model serving and infernece in any Java based frameworks (e.g., [Spring Framework](https://spring.io), Apache [Storm](http://storm.apache.org), [Kafka](http://kafka.apache.org) or [Flink](http://flink.apache.org), etc.), as illustrated below:

```python
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
}

TextClassificationModel model = new TextClassificationModel();
model.load(modelPath, weightPath);

List<JTensor> inputs = preprocess(...);
List<List<JTensor>> result = model.predict(inputs);
...
```

## _Built-in deep learning models_
Analytics Zoo provides several built-in deep learning models that you can use for a variety of problem types, such as *object detection*, *image classification*, *text classification*, *recommendation*, *anomaly detection*, *text matching*, etc.

### _Object detection API_
Using *Analytics Zoo Object Detection API* (including a set of pretrained detection models such as SSD and Faster-RCNN), you can easily build your object detection applications (e.g., localizing and identifying multiple objects in images and videos), as illustrated below. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/object-detection/))

1. Download object detection models in Analytics Zoo

   You can download a collection of detection models (pretrained on the PSCAL VOC dataset and COCO dataset) from [detection model zoo](https://analytics-zoo.github.io/master/#ProgrammingGuide/object-detection/#download-link).

2. Use *Object Detection API* for off-the-shell inference
   ```python
   from zoo.models.image.objectdetection import *
   model = ObjectDetector.load_model(model_path)
   image_set = ImageSet.read(img_path, sc)
   output = model.predict_image_set(image_set)
   ```

### _Image classification API_
Using *Analytics Zoo Image Classification API* (including a set of pretrained detection models such as VGG, Inception, ResNet, MobileNet,  etc.), you can easily build your image classification applications, as illustrated below. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/))

1. Download image classification models in Analytics Zoo

   You can download a collection of image classification models (pretrained on the ImageNet dataset) from [image classification model zoo](https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/#download-link).

2. Use *Image classification API* for off-the-shell inference
   ```python
   from zoo.models.image.imageclassification import *
   model = ImageClassifier.load_model(model_path)
   image_set = ImageSet.read(img_path, sc)
   output = model.predict_image_set(image_set)
   ```

### _Text classification API_
*Analytics Zoo Text Classification API* provides a set of pre-defined models (using CNN, LSTM, etc.) for text classifications. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/text-classification/))

### _Recommendation API_
*Analytics Zoo Recommendation API* provides a set of pre-defined models (such as Neural Collaborative Filtering, Wide and Deep Learning, etc.) for recommendations. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/recommendation/))

### _Anomaly Detection API_
*Analytics Zoo Anomaly Detection API* provides a set of pre-defined models based on LSTM to detect anomalies for time series data. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/anomaly-detection/))

### _Text matching API_
*Analytics Zoo Text Matching API* provides pre-defined KNRM model for ranking or classification. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/text-matching/))

## _Reference use cases_
Analytics Zoo provides a collection of end-to-end reference use cases, including *time series anomaly detection*, *sentiment analysis*, *fraud detection*, *image similarity*, etc. (See more details [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/usercases-overview/))
