Analytics Zoo provides a collection of pre-trained models for Object Detection. These models can be used for out-of-the-box inference if you are interested in categories already in the corresponding datasets. According to the business scenarios. User can run the inference as local program without Spark Context, or in a distributed environment such like Apache Spark, Apache Storm or Apache Flink.

## Model Load

Use `ObjectDetector.loadModel`(in Scala) or `ObjectDetector.load_model` (in Python) to load an pre-trained Analytics Zoo model or third-party(BigDL) model. We just need to specify the model path and optionally weight path if exists where we previously saved the model.

**Scala example**
```scala
import com.intel.analytics.zoo.models.image.objectdetection._

val model = ObjectDetector.loadModel[Float]("/tmp/zoo.model") //load from local fs
val model = ObjectDetector.loadModel("hdfs://...") //load from hdfs
val model = ObjectDetector.loadModel("s3://...") //load from s3
```

**Python example**
```python
from zoo.models.image.objectdetection import *

model = ObjectDetector.load_model("/tmp/zoo.model") //load from local fs
model = ObjectDetector.load_model("hdfs://...") //load from hdfs
model = ObjectDetector.load_model("s3://...") //load from s3
```

## Creat image configuration
If the loaded model is a published Analytics Zoo model, when you call `ObjectDetector.loadModel`(in Scala) or `ObjectDetector.load_model` (in Python), it would create the default Image Configuration for model inference. If the loaded model is not a published Analytics Zoo model or you want to customize the configuration for model inference, you need to create your own Image Configuration.

**Scala API**
```
ImageConfigure[T: ClassTag](
  preProcessor: Preprocessing[ImageFeature, ImageFeature] = null,
  postProcessor: Preprocessing[ImageFeature, ImageFeature] = null,
  batchPerPartition: Int = 4,
  labelMap: Map[Int, String] = null,
  featurePaddingParam: Option[PaddingParam[T]] = None)
```
* preProcessor: preprocessor of ImageSet before model inference
* postProcessor: postprocessor of ImageSet after model inference
* batchPerPartition: batch size per partition
* labelMap: label mapping
* featurePaddingParam: featurePaddingParam if the inputs have variant size

**Scala example**
```scala
import com.intel.analytics.zoo.models.image.common._
import com.intel.analytics.zoo.feature.image._

val preprocessing = ImageResize(256, 256)-> ImageCenterCrop(224, 224) ->
                     ImageChannelNormalize(123, 117, 104) ->
                     ImageMatToTensor[Float]() ->
                     ImageSetToSample[Float]()
val config = ImageConfigure[Float](preProcessor=preprocessing)
```


**Python API**
```
class ImageConfigure()
    def __init__(self, pre_processor=None,
                 post_processor=None,
                 batch_per_partition=4,
                 label_map=None, feature_padding_param=None, jvalue=None, bigdl_type="float")
```
* pre_processor:  preprocessor of ImageSet before model inference
* post_processor:  postprocessor of ImageSet after model inference
* batch_per_partition:  batch size per partition
* label_map mapping:  from prediction result indexes to real dataset labels
* feature_padding_param:  featurePaddingParam if the inputs have variant size

**Python example**
```python
from zoo.models.image.common.image_config import *
from zoo.feature.image.imagePreprocessing import *

preprocessing = ChainedPreprocessing(
                [ImageResize(256, 256), ImageCenterCrop(224, 224),
                ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(),
                ImageSetToSample()])
config = ImageConfigure(pre_processor=preprocessing)
```

## Predict with loaded object detection model

**Scala API**
```
predictImageSet(image: ImageSet, configure: ImageConfigure[T] = null)
```
* image:  Analytics Zoo ImageSet to be predicted
* configure: Image Configure for this prediction

**Scala example**
```scala
import com.intel.analytics.zoo.models.image.objectdetection._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._

val imagePath="/tmp/image"
val sc = NNContext.initNNContext()
val model = ObjectDetector.loadModel("/tmp/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model")
val data = ImageSet.read(image_path, sc)
val output = model.predictImageSet(data)
```


**Python API**
```
predict_image_set(image, configure=None)
```
* image:  Analytics Zoo ImageSet to be predicted
* configure: Image Configure for this  prediction

**Python example**
```python
from zoo.common.nncontext import *
from zoo.models.image.objectdetection import *

sc = init_nncontext()
model = ObjectDetector.load_model(model_path)
image_set = ImageSet.read(img_path, sc)
output = model.predict_image_set(image_set)
```
