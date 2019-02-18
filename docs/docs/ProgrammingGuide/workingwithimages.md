Analytics Zoo provides supports for end-to-end image processing pipeline, including image loading, pre-processing, inference/training and some utilities on different formats.

## Load Image
Analytics Zoo provides APIs to read image to different formats:

### Load to Data Frame
Analytics Zoo can process image data as Spark Data Frame.
`NNImageReader` is the primary DataFrame-based image loading interface to read images into DataFrame.

Scala example:
```scala
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.NNImageReader

val sc = NNContext.initNNContext("app")
val imageDF1 = NNImageReader.readImages("/tmp", sc)
val imageDF2 = NNImageReader.readImages("/tmp/*.jpg", sc)
val imageDF3 = NNImageReader.readImages("/tmp/a.jpg, /tmp/b.jpg", sc)

```

Python:
```python
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *

sc = init_nncontext("app")
imageDF1 = NNImageReader.readImages("/tmp", sc)
imageDF2 = NNImageReader.readImages("/tmp/*.jpg", sc)
imageDF3 = NNImageReader.readImages("/tmp/a.jpg, /tmp/b.jpg", sc)
```

The output DataFrame contains a sinlge column named "image". The schema of "image" column can be
accessed from `com.intel.analytics.zoo.pipeline.nnframes.DLImageSchema.byteSchema`.
Each record in "image" column represents one image record, in the format of
Row(origin, height, width, num of channels, mode, data), where origin contains the URI for the image file,
and `data` holds the original file bytes for the image file. `mode` represents the OpenCV-compatible
type: CV_8UC3, CV_8UC1 in most cases.
```scala
  val byteSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_8UC3, CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // Bytes in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", BinaryType, false) :: Nil)
```

After loading the image, user can compose the preprocess steps with the `Preprocessing` defined
in `com.intel.analytics.zoo.feature.image`.

### Load to ImageSet
`ImageSet` is a collection of `ImageFeature`. It can be a `DistributedImageSet` for distributed image RDD or
 `LocalImageSet` for local image array.
You can read image data as `ImageSet` from local/distributed image path, or you can directly construct a ImageSet from RDD[ImageFeature] or Array[ImageFeature].

**Scala example:**

```scala
// create LocalImageSet from an image folder
val localImageSet = ImageSet.read("/tmp/image/")

// create DistributedImageSet from an image folder
val distributedImageSet2 = ImageSet.read("/tmp/image/", sc, 2)
```

**Python example:**

```python
# create LocalImageSet from an image folder
local_image_frame2 = ImageSet.read("/tmp/image/")

# create DistributedImageSet from an image folder
distributed_image_frame = ImageSet.read("/tmp/image/", sc, 2)
```

## Image Transformer
Analytics Zoo has many pre-defined image processing transformers built on top of OpenCV:

* `ImageBrightness`: Adjust the image brightness.
* `ImageHue`: Adjust the image hue.
* `ImageSaturation`: Adjust the image Saturation.
* `ImageContrast`: Adjust the image Contrast.
* `ImageChannelOrder`: Random change the channel order of an image
* `ImageColorJitter`: Random adjust brightness, contrast, hue, saturation
* `ImageResize`: Resize image
* `ImageAspectScale`: Resize the image, keep the aspect ratio. scale according to the short edge
* `ImageRandomAspectScale`: Resize the image by randomly choosing a scale
* `ImageChannelNormalize`: Image channel normalize
* `ImagePixelNormalizer`: Pixel level normalizer
* `ImageCenterCrop`: Crop a `cropWidth` x `cropHeight` patch from center of image.
* `ImageRandomCrop`: Random crop a `cropWidth` x `cropHeight` patch from an image.
* `ImageFixedCrop`: Crop a fixed area of image
* `ImageDetectionCrop`: Crop from object detections, each image should has a tensor detection,
* `ImageExpand`: Expand image, fill the blank part with the meanR, meanG, meanB
* `ImageFiller`: Fill part of image with certain pixel value
* `ImageHFlip`: Flip the image horizontally
* `ImageRandomPreprocessing`: It is a wrapper for transformers to control the transform probability
* `ImageBytesToMat`: Transform byte array(original image file in byte) to OpenCVMat
* `ImageMatToFloats`: Transform OpenCVMat to float array, note that in this transformer, the mat is released.
* `ImageMatToTensor`: Transform opencv mat to tensor, note that in this transformer, the mat is released.
* `ImageSetToSample`: Transforms tensors that map inputKeys and targetKeys to sample, note that in this transformer, the mat has been released.

More examples can be found [here](../APIGuide/FeatureEngineering/image.md)

You can also define your own Transformer by extending `ImageProcessing`,
and override the function `transformMat` to do the actual transformation to `ImageFeature`.

## **Build Image Transformation Pipeline**
You can easily build the image transformation pipeline by chaining transformers.

**Scala example:**

```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.feature.image._


val imgAug = ImageBytesToMat() -> ImageResize(256, 256)-> ImageCenterCrop(224, 224) ->
             ImageChannelNormalize(123, 117, 104) ->
             ImageMatToTensor[Float]() ->
             ImageSetToSample[Float]()
```
In the above example, the transformations will perform sequentially.

Assume you have an ImageSet containing original bytes array,

* `ImageBytesToMat` will transform the bytes array to `OpenCVMat`.

* `ImageColorJitter`, `ImageExpand`, `ImageResize`, `ImageHFlip` and `ImageChannelNormalize` will transform over `OpenCVMat`,
note that `OpenCVMat` is overwrite by default.

* `ImageMatToTensor` transform `OpenCVMat` to `Tensor`, and `OpenCVMat` is released in this step.

* `ImageSetToSample` transform the tensors that map inputKeys and targetKeys to sample,
which can be used by the following prediction or training tasks.

**Python example:**

```python
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.common import ChainedPreprocessing

img_aug = ChainedPreprocessing([ImageBytesToMat(),
      ImageColorJitter(),
      ImageExpand(),
      ImageResize(300, 300, -1),
      ImageHFlip(),
      ImageChannelNormalize(123.0, 117.0, 104.0),
      ImageMatToTensor(),
      ImageSetToSample()])
```

## **Image Train**
### Train with Image DataFrame
You can use NNEstimator/NNCLassifier to train Zoo Keras/BigDL model with Image DataFrame. You can pass in image preprocessing to NNEstimator/NNClassifier to do image preprocessing before training. Then call `fit` method to let Analytics Zoo train the model

For detail APIs, please refer to: [NNFrames](../APIGuide/PipelineAPI/nnframes.md)

**Scala example:**
```scala
val batchsize = 128
val nEpochs = 10
val featureTransformer = RowToImageFeature() -> ImageResize(256, 256) ->
                                   ImageCenterCrop(224, 224) ->
                                   ImageChannelNormalize(123, 117, 104) ->
                                   ImageMatToTensor() ->
                                   ImageFeatureToTensor()
val classifier = NNClassifier(model, CrossEntropyCriterion[Float](), featureTransformer)
        .setFeaturesCol("image")
        .setLearningRate(0.003)
        .setBatchSize(batchsize)
        .setMaxEpoch(nEpochs)
        .setValidation(Trigger.everyEpoch, valDf, Array(new Top1Accuracy()), batchsize)
val trainedModel = classifier.fit(trainDf)
```
**Python example:**
```python
batchsize = 128
nEpochs = 10
featureTransformer = ChainedPreprocessing([RowToImageFeature(), ImageResize(256, 256),
                                   ImageCenterCrop(224, 224),
                                   ImageChannelNormalize(123, 117, 104),
                                   ImageMatToTensor(),
                                   ImageFeatureToTensor()])
classifier = NNClassifier(model, CrossEntropyCriterion(), featureTransformer)\
        .setFeaturesCol("image")\
        .setLearningRate(0.003)\
        .setBatchSize(batchsize)\
        .setMaxEpoch(nEpochs)\
        .setValidation(EveryEpoch(), valDf, [Top1Accuracy()], batch_size)
trainedModel = classifier.fit(trainDf)
```
### Train with ImageSet

You can train Zoo Keras model with ImageSet. Just call `fit` method to let Analytics Zoo train the model.

**Python example:**

```python
from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import *
from bigdl.optim.optimizer import *

sc = init_nncontext("train keras")
img_path="/tmp/image"
image_set = ImageSet.read(img_path,sc, min_partitions=1)
transformer = ChainedPreprocessing(
        [ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(),
         ImageSetToSample()])
image_data = transformer(image_set)
labels = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
label_rdd = sc.parallelize(labels, 1)
samples = image_data.get_image().zip(label_rdd).map(
        lambda tuple: Sample.from_ndarray(tuple[0], tuple[1]))
# create model
model_path="/tmp/bigdl_inception-v1_imagenet_0.4.0.model"
full_model = Net.load_bigdl(model_path)
# create a new model by remove layers after pool5/drop_7x7_s1
model = full_model.new_graph(["pool5/drop_7x7_s1"])
# freeze layers from input to pool4/3x3_s2 inclusive
model.freeze_up_to(["pool4/3x3_s2"])

inputNode = Input(name="input", shape=(3, 224, 224))
inception = model.to_keras()(inputNode)
flatten = Flatten()(inception)
logits = Dense(2)(flatten)
lrModel = Model(inputNode, logits)

batchsize = 4
nEpochs = 10
lrModel.compile(optimizer=Adam(learningrate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
lrModel.fit(x = samples, batch_size=batchsize, nb_epoch=nEpochs)
```

## **Image Predict**
### Predict with Image DataFrame
After training with *NNEstimator/NNCLassifier*, you'll get a trained *NNModel/NNClassifierModel* . You can call `transform` to predict Image DataFrame with this *NNModel/NNClassifierModel* . Or you can load pre-trained *Analytics-Zoo/BigDL/Caffe/Torch/Tensorflow*  model and create *NNModel/NNClassifierModel* with this model. Then call to `transform` to Image DataFrame.

After prediction, there is a new column `prediction` in the prediction image dataframe.
 
 **Scala example:**
 
```scala
 val batchsize = 128
 val nEpochs = 10
 val featureTransformer = RowToImageFeature() -> ImageResize(256, 256) ->
                                    ImageCenterCrop(224, 224) ->
                                    ImageChannelNormalize(123, 117, 104) ->
                                    ImageMatToTensor() ->
                                    ImageFeatureToTensor()
 val classifier = NNClassifier(model, CrossEntropyCriterion[Float](), featureTransformer)
         .setFeaturesCol("image")
         .setLearningRate(0.003)
         .setBatchSize(batchsize)
         .setMaxEpoch(nEpochs)
         .setValidation(Trigger.everyEpoch, valDf, Array(new Top1Accuracy()), batchsize)
 val trainedModel = classifier.fit(trainDf)
 // predict with trained model
 val predictions = trainedModel.transform(testDf)
 predictions.select(col("image"), col("label"), col("prediction")).show(false)
 
 // predict with loaded pre-trained model
 val model = Module.loadModule[Float](modelPath)
 val dlmodel = NNClassifierModel(model, featureTransformer)
         .setBatchSize(batchsize)
         .setFeaturesCol("image")
         .setPredictionCol("prediction") 
 val resultDF = dlmodel.transform(testDf)
```
 
 **Python example:**

```python
 batchsize = 128
 nEpochs = 10
 featureTransformer = ChainedPreprocessing([RowToImageFeature(), ImageResize(256, 256),
                                    ImageCenterCrop(224, 224),
                                    ImageChannelNormalize(123, 117, 104),
                                    ImageMatToTensor(),
                                    ImageFeatureToTensor()])
 classifier = NNClassifier(model, CrossEntropyCriterion(), featureTransformer)\
         .setFeaturesCol("image")\
         .setLearningRate(0.003)\
         .setBatchSize(batchsize)\
         .setMaxEpoch(nEpochs)\
         .setValidation(EveryEpoch(), valDf, [Top1Accuracy()], batch_size)
trainedModel = classifier.fit(trainDf)
# predict with trained model
predictions = trainedModel.transform(testDf)
predictions.select("image", "label","prediction").show(False)

# predict with loaded pre-trained model
model = Model.loadModel(model_path)
dlmodel = NNClassifierModel(model, featureTransformer)\
         .setBatchSize(batchsize)\
         .setFeaturesCol("image")\
         .setPredictionCol("prediction") 
resultDF = dlmodel.transform(testDf)
```
### Predict with ImageSet

After training Zoo Keras model, you can call `predict` to predict ImageSet.
Or you can load pre-trained Analytics-Zoo/BigDL model. Then call to `predictImageSet` to predict ImageSet.

#### Predict with trained Zoo Keras Model

**Python example:**

```python
from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import *
from bigdl.optim.optimizer import *

sc = init_nncontext("train keras")
img_path="/tmp/image"
image_set = ImageSet.read(img_path,sc, min_partitions=1)
transformer = ChainedPreprocessing(
        [ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(),
         ImageSetToSample()])
image_data = transformer(image_set)
labels = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
label_rdd = sc.parallelize(labels, 1)
samples = image_data.get_image().zip(label_rdd).map(
        lambda tuple: Sample.from_ndarray(tuple[0], tuple[1]))
# create model
model_path="/tmp/bigdl_inception-v1_imagenet_0.4.0.model"
full_model = Net.load_bigdl(model_path)
# create a new model by remove layers after pool5/drop_7x7_s1
model = full_model.new_graph(["pool5/drop_7x7_s1"])
# freeze layers from input to pool4/3x3_s2 inclusive
model.freeze_up_to(["pool4/3x3_s2"])

inputNode = Input(name="input", shape=(3, 224, 224))
inception = model.to_keras()(inputNode)
flatten = Flatten()(inception)
logits = Dense(2)(flatten)
lrModel = Model(inputNode, logits)

batchsize = 4
nEpochs = 10
lrModel.compile(optimizer=Adam(learningrate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
lrModel.fit(x = samples, batch_size=batchsize, nb_epoch=nEpochs)
prediction = lrModel.predict(samples)
result = prediction.collect()
``` 

#### Predict with loaded Model
You can load pre-trained Analytics-Zoo/BigDL model. Then call to `predictImageSet` to predict ImageSet.

For details, you can check guide of [image classificaion](./image-classification.md) or [object detection](./object-detection.md)

## 3D Image Support
For 3D images, we can support above operations based on ImageSet. For details, please refer to [image API guide](../APIGuide/FeatureEngineering/image.md)

## Caching Images in Persistent Memory
Here is a scala [example](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception/README.md) to train Inception V1 with ImageNet-2012 dataset. If you set the option `memoryType` to `PMEM`, the data will be cached in Intel Optane DC Persistent Memory; please refer to the guide [here](https://github.com/memkind/memkind#run-requirements) on how to set up the system environment.

In the InceptionV1 example, we use an new dataset called [FeatureSet](../APIGuide/FeatureEngineering/featureset.md) to cache the data. Only scala API is currently available.

 **Scala example:**
 
 ```scala
 val rawData = readFromSeqFiles(path, sc, classNumber)
 val featureSet = FeatureSet.rdd(rawData, memoryType = PMEM)
 ```
 `readFromSeqFiles` read the Sequence File into `RDD[ByteRecord]`, then `FeatureSet.rdd(rawData, memoryType = PMEM)` will cache the data to Intel Optane DC Persistent Memory.

