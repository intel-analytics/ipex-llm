## **Overview**

BigDL provides supports for end-to-end image processing pipeline,
including image loading, pre-processing, inference/training and some utilities.

The basic unit of an image is `ImageFeature`, which describes various status of the image
by using key-value store.
For example, `ImageFeature` can include original image file in bytes, image in OpenCVMat format,
image uri, image meta data and so on.

`ImageFrame` is a collection of `ImageFeature`.
It can be a `DistributedImageFrame` for distributed image RDD or
 `LocalImageFrame` for local image array.

## **Image Loading**

You can read an `ImageFrame` from local/distributed folder/parquet file,
or you can directly construct a ImageFrame from RDD[ImageFeature] or Array[ImageFeature].

**Scala example:**

```scala
// create LocalImageFrame from an image folder
val localImageFrame = ImageFrame.read("/tmp/image/")

// create DistributedImageFrame from an image folder
val distributedImageFrame2 = ImageFrame.read("/tmp/image/", sc, 2)
```

**Python example:**

```python
# create LocalImageFrame from an image folder
local_image_frame2 = ImageFrame.read("/tmp/image/")

# create DistributedImageFrame from an image folder
distributed_image_frame = ImageFrame.read("/tmp/image/", sc, 2)
```

More examples can be found [here](../APIGuide/Data.md#imageframe)


## **Image Transformer**
BigDL has many pre-defined image transformers built on top of OpenCV:

* `Brightness`: Adjust the image brightness.
* `Hue`: Adjust the image hue.
* `Saturation`: Adjust the image Saturation.
* `Contrast`: Adjust the image Contrast.
* `ChannelOrder`: Random change the channel order of an image
* `ColorJitter`: Random adjust brightness, contrast, hue, saturation
* `Resize`: Resize image
* `AspectScale`: Resize the image, keep the aspect ratio. scale according to the short edge
* `RandomAspectScale`: Resize the image by randomly choosing a scale
* `ChannelNormalize`: Image channel normalize
* `PixelNormalizer`: Pixel level normalizer
* `CenterCrop`: Crop a `cropWidth` x `cropHeight` patch from center of image.
* `RandomCrop`: Random crop a `cropWidth` x `cropHeight` patch from an image.
* `FixedCrop`: Crop a fixed area of image
* `DetectionCrop`: Crop from object detections, each image should has a tensor detection,
* `Expand`: Expand image, fill the blank part with the meanR, meanG, meanB
* `Filler`: Fill part of image with certain pixel value
* `HFlip`: Flip the image horizontally
* `RandomTransformer`: It is a wrapper for transformers to control the transform probability
* `BytesToMat`: Transform byte array(original image file in byte) to OpenCVMat
* `MatToFloats`: Transform OpenCVMat to float array, note that in this transformer, the mat is released.
* `MatToTensor`: Transform opencv mat to tensor, note that in this transformer, the mat is released.
* `ImageFrameToSample`: Transforms tensors that map inputKeys and targetKeys to sample, note that in this transformer, the mat has been released.

More examples can be found [here](../APIGuide/Transformer.md)

You can also define your own Transformer by extending `FeatureTransformer`,
and override the function `transformMat` to do the actual transformation to `ImageFeature`.

## **Build Image Transformation Pipeline**
You can easily build the image transformation pipeline by chaining transformers.

**Scala example:**

```scala
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._

val imgAug = BytesToMat() -> ColorJitter() ->
      Expand() ->
      Resize(300, 300, -1) ->
      HFlip() ->
      ChannelNormalize(123, 117, 104) ->
      MatToTensor() -> ImageFrameToSample()
```
In the above example, the transformations will perform sequentially.

Assume you have an ImageFrame containing original bytes array,
`BytesToMat` will transform the bytes array to `OpenCVMat`.

`ColorJitter`, `Expand`, `Resize`, `HFlip` and `ChannelNormalize` will transform over `OpenCVMat`,
note that `OpenCVMat` is overwrite by default.

`MatToTensor` transform `OpenCVMat` to `Tensor`, and `OpenCVMat` is released in this step.

`ImageFrameToSample` transform the tensors that map inputKeys and targetKeys to sample,
which can be used by the following prediction or training tasks.

**Python example:**

```python
from bigdl.util.common import *
from bigdl.transform.vision.image import *

img_aug = Pipeline([BytesToMat(),
      ColorJitter(),
      Expand(),
      Resize(300, 300, -1),
      HFlip(),
      ChannelNormalize(123.0, 117.0, 104.0),
      MatToTensor(),
      ImageFrameToSample()])
```

## **Image Prediction**
BigDL provides easy-to-use prediction API `predictImage` for `ImageFrame`.

**Scala:**
```scala
model.predictImage(imageFrame: ImageFrame,
                   outputLayer: String = null,
                   shareBuffer: Boolean = false,
                   batchPerPartition: Int = 4,
                   predictKey: String = ImageFeature.predict)
```
**Python:**
```python
model.predict_image(image_frame, output_layer=None, share_buffer=False,
                    batch_per_partition=4, predict_key="predict")
```
Model predict images, return imageFrame with predicted tensor

   * `imageFrame` imageFrame that contains images
   * `outputLayer` if outputLayer is not null, the output of layer that matches outputLayer will be used as predicted output
   * `shareBuffer` whether to share same memory for each batch predict results
   * `batchPerPartition` batch size per partition, default is 4
   * `predictKey` key to store predicted result

## **Construct Image Prediction Pipeline**

With the above image-related supports, we can easily build a image prediction pipeline.

**Scala example:**


```scala
val imageFrame = ImageFrame.read(imagePath, sc, nPartition)
val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
                 ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
                 MatToTensor() -> ImageFrameToSample()
val transformed = transformer(imageFrame)
val model = Module.loadModule(modelPath)
val output = model.predictImage(transformed)
```
The above example read a distributed ImageFrame, and performs data pre-processing.
Then it loads a pre-trained BigDL model, and predicts over imageFrame.
It returns imageFrame with prediction result, which can be accessed by the key `ImageFeature.predict`.

If you want to run the local example, just replace `ImageFrame.read(imagePath, sc, nPartition)`
with `ImageFrame.read(imagePath)`.

**Python example:**

```python
image_frame = ImageFrame.read(image_path, self.sc)
transformer = Pipeline([Resize(256, 256), CenterCrop(224, 224),
                        ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                        MatToTensor(), ImageFrameToSample()])
transformed = transformer(image_frame)
model = Model.loadModel(model_path)
output = model.predict_image(image_frame)
```

You can call `output.get_predict()` to get the prediction results.
