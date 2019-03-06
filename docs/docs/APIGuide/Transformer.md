Transformer is for pre-processing. In many deep learning workload, input data need to be pre-processed before fed into   model. For example, in CNN, the image file need to be decoded from some compressed format(e.g. jpeg) to float arrays,    normalized and cropped to some fixed shape. You can also find pre-processing in other types of deep learning work        load(e.g. NLP, speech recognition). In BigDL, we provide many pre-process procedures for user. They're implemented as    Transformer.

The transformer interface is
```scala

trait Transformer[A, B] extends Serializable {
   def apply(prev: Iterator[A]): Iterator[B]
 }
```

It's simple, right? What a transformer do is convert a sequence of objects of Class A to a sequence of objects of Class  B.

Transformer is flexible. You can chain them together to do pre-processing. Let's still use the CNN example, say first    we need read image files from given paths, then extract the image binaries to array of float, then normalize the image  content and crop a fixed size from the image at a random position. Here we need 4 transformers, `PathToImage`,           `ImageToArray`, `Normalizor` and `Cropper`. And then chain them together.

## **FeatureTransformer**
`FeatureTransformer` is the transformer that transforms from `ImageFeature` to `ImageFeature`.
`FeatureTransformer` extends 'Transformer[ImageFeature, ImageFeature]'.

FeatureTransformer can be chained with FeatureTransformer with the

The key function in `FeatureTransformer` is `transform`, which does the ImageFeature transformation
and exception control.
While `transformMat` is called by `transform`,
and it is expected to contain the actual transformation of an ImageFeature.
It is advised to override `transformMat` when you implement your own FeatureTransformer.

---
## **Brightness**
**Scala:**
```scala
val brightness = Brightness(deltaLow: Double, deltaHigh: Double)
```
**Python:**
```python
brightness = Brightness(delta_low, delta_high)
```
Adjust the image brightness.
* `deltaLow` brightness parameter: low bound
* `deltaHigh` brightness parameter: high bound

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Brightness(0, 32)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
brightness = Brightness(0.0, 32.0)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = brightness(local_image_frame)
```

---
## **Hue**
**Scala:**
```scala
val transformer = Hue(deltaLow: Double, deltaHigh: Double)
```
**Python:**
```python
transformer = Hue(delta_low, delta_high)
```
Adjust the image hue.
* `deltaLow` Hue parameter: low bound
* `deltaHigh` Hue parameter: high bound

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Hue(-18, 18)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = Hue(-18.0, 18.0)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **Saturation**
**Scala:**
```scala
val transformer = Saturation(deltaLow: Double, deltaHigh: Double)
```
**Python:**
```python
transformer = Saturation(delta_low, delta_high)
```
Adjust the image Saturation.
* `deltaLow` Saturation parameter: low bound
* `deltaHigh` Saturation parameter: high bound

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Saturation(10, 20)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = Saturation(10.0, 20.0)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **Contrast**
**Scala:**
```scala
val transformer = Contrast(deltaLow: Double, deltaHigh: Double)
```
**Python:**
```python
transformer = Contrast(delta_low, delta_high)
```
Adjust the image Contrast.
* `deltaLow` Contrast parameter: low bound
* `deltaHigh` Contrast parameter: high bound

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Contrast(0.5, 1.5)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = Hue(0.5, 1.5)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **ChannelOrder**
**Scala:**
```scala
val transformer = ChannelOrder()
```
**Python:**
```python
transformer = ChannelOrder()
```
Random change the channel order of an image

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = ChannelOrder()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = ChannelOrder()
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **ColorJitter**
**Scala:**
```scala
val transformer = ColorJitter(brightnessProb: Double = 0.5,
                              brightnessDelta: Double = 32,
                              contrastProb: Double = 0.5,
                              contrastLower: Double = 0.5,
                              contrastUpper: Double = 1.5,
                              hueProb: Double = 0.5,
                              hueDelta: Double = 18,
                              saturationProb: Double = 0.5,
                              saturationLower: Double = 0.5,
                              saturationUpper: Double = 1.5,
                              randomOrderProb: Double = 0,
                              shuffle: Boolean = false)
```
**Python:**
```python
transformer = ColorJitter(brightness_prob = 0.5,
                           brightness_delta = 32.0,
                           contrast_prob = 0.5,
                           contrast_lower = 0.5,
                           contrast_upper = 1.5,
                           hue_prob = 0.5,
                           hue_delta = 18.0,
                           saturation_prob = 0.5,
                           saturation_lower = 0.5,
                           saturation_upper = 1.5,
                           random_order_prob = 0.0,
                           shuffle = False)
```
Random adjust brightness, contrast, hue, saturation

 * `brightnessProb`: probability to adjust brightness
 * `brightnessDelta`: brightness parameter
 * `contrastProb`: probability to adjust contrast
 * `contrastLower`: contrast lower parameter
 * `contrastUpper`: contrast upper parameter
 * `hueProb`: probability to adjust hue
 * `hueDelta`: hue parameter
 * `saturationProb`: probability to adjust saturation
 * `saturationLower`: saturation lower parameter
 * `saturationUpper`: saturation upper parameter
 * `randomChannelOrderProb`: random order for different operation
 * `shuffle`: shuffle the transformers

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = ColorJitter()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = ColorJitter()
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **Resize**
**Scala:**
```scala
val transformer = Resize(resizeH: Int, resizeW: Int,
                    resizeMode: Int = Imgproc.INTER_LINEAR,
                    useScaleFactor: Boolean = true)
```
**Python:**
```python
transformer = Resize(resize_h, resize_w, resize_mode = 1, use_scale_factor=True)
```
Resize image
 * `resizeH` height after resize
 * `resizeW` width after resize
 * `resizeMode` if resizeMode = -1, random select a mode from
(Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
                   Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
 * `useScaleFactor` if true, scale factor fx and fy is used, fx = fy = 0
 note that the result of the following are different:

```python
Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH), 0, 0, Imgproc.INTER_LINEAR)
Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH))
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Resize(300, 300)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = Resize(300, 300)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **AspectScale**
**Scala:**
```scala
val transformer = AspectScale(scale: Int, scaleMultipleOf: Int = 1,
                    maxSize: Int = 1000)
```
**Python:**
```python
transformer = AspectScale(scale, scale_multiple_of = 1, max_size = 1000)
```
Resize the image, keep the aspect ratio. scale according to the short edge
 * `scale` scale size, apply to short edge
 * `scaleMultipleOf` make the scaled size multiple of some value
 * `maxSize` max size after scale


**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = AspectScale(750, maxSize = 3000)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = AspectScale(750, max_size = 3000)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **RandomAspectScale**
**Scala:**
```scala
val transformer = AspectScale(scale: Int, scaleMultipleOf: Int = 1,
                    maxSize: Int = 1000)
```
**Python:**
```python
transformer = AspectScale(scale, scale_multiple_of = 1, max_size = 1000)
```
resize the image by randomly choosing a scale
 * `scales` array of scale options that for random choice
 * `scaleMultipleOf` Resize test images so that its width and height are multiples of
 * `maxSize` Max pixel size of the longest side of a scaled input image

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = RandomAspectScale(Array(750, 600), maxSize = 3000)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = RandomAspectScale([750, 600], max_size = 3000)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **ChannelNormalize**
**Scala:**
```scala
val transformer = ChannelNormalize(meanR: Float, meanG: Float, meanB: Float,
                                         stdR: Float = 1, stdG: Float = 1, stdB: Float = 1)
```
**Python:**
```python
transformer = ChannelNormalize(mean_r, mean_b, mean_g, std_r=1.0, std_g=1.0, std_b=1.0)
```
image channel normalize
 * `meanR` mean value in R channel
 * `meanG` mean value in G channel
 * `meanB` mean value in B channel
 * `stdR` std value in R channel
 * `stdG` std value in G channel
 * `stdB` std value in B channel

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = ChannelNormalize(100f, 200f, 300f, 2f, 3f, 4f)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = ChannelNormalize(100.0, 200.0, 300.0, 2.0, 3.0, 4.0)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```
---
## **PixelNormalizer**
**Scala:**
```scala
val transformer = PixelNormalizer(means: Array[Float])
```
**Python:**
```python
transformer = PixelNormalizer(means)
```
Pixel level normalizer, data(i) = data(i) - mean(i)

 * `means` pixel level mean, following H * W * C order

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
// Assume the image pixels length is 375 * 500 * 3
val means = new Array[Float](375 * 500 * 3)
val transformer = PixelNormalizer(means)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
means = [2.0] * 3 * 500 * 375
transformer = PixelNormalize(means)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **CenterCrop**
**Scala:**
```scala
val transformer = CenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean = true)
```
**Python:**
```python
transformer = CenterCrop(crop_width, crop_height, is_clip=True)
```
Crop a `cropWidth` x `cropHeight` patch from center of image.
The patch size should be less than the image size.

 * `cropWidth` width after crop
 * `cropHeight` height after crop
 * `isClip` whether to clip the roi to image boundaries

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = CenterCrop(200, 200)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = CenterCrop(200, 200)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **RandomCrop**
**Scala:**
```scala
val transformer = RandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean = true)
```
**Python:**
```python
transformer = RandomCrop(crop_width, crop_height, is_clip=True)
```
Random crop a `cropWidth` x `cropHeight` patch from an image.
The patch size should be less than the image size.

 * `cropWidth` width after crop
 * `cropHeight` height after crop
 * `isClip` whether to clip the roi to image boundaries

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = RandomCrop(200, 200)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
transformer = RandomCrop(200, 200)
local_image_frame = ImageFrame.read("/tmp/test.jpg")
transformed = transformer(local_image_frame)
```

---
## **FixedCrop**
**Scala:**
```scala
val transformer = FixedCrop(x1: Float, y1: Float, x2: Float, y2: Float, normalized: Boolean,
                      isClip: Boolean = true)
```
**Python:**
```python
transformer = FixedCrop(x1, y1, x2, y2, normalized=True, is_clip=True)
```
Crop a fixed area of image

 * `x1` start in width
 * `y1` start in height
 * `x2` end in width
 * `y2` end in height
 * `normalized` whether args are normalized, i.e. in range [0, 1]
 * `isClip` whether to clip the roi to image boundaries

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
val data = ImageFrame.read("/tmp/test.jpg")
val transformer = FixedCrop(0, 0, 50, 50, false)
val transformed = transformer(data)

val transformer2 = FixedCrop(0, 0, 0.1f, 0.1333f, true)
val transformed2 = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
local_image_frame = ImageFrame.read("/tmp/test.jpg")

transformer = FixedCrop(0.0, 0.0, 50.0, 50.0, False)
transformed = transformer(local_image_frame)

transformer2 = FixedCrop(0.0, 0.0, 0.1, 0.1333, True)
transformed2 = transformer(local_image_frame)
```

---
## **DetectionCrop**
**Scala:**
```scala
val transformer = DetectionCrop(roiKey: String, normalized: Boolean = true)
```
**Python:**
```python
transformer = DetectionCrop(roi_key, normalized=True)
```
Crop from object detections, each image should has a tensor detection,
which is stored in ImageFeature

 * `roiKey` roiKey that map a tensor detection
 * `normalized` whether is detection is normalized, i.e. in range [0, 1]

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

val data = ImageFrame.read("/tmp/test.jpg").toLocal()
val imf = data.array(0)
imf("roi") = Tensor[Float](T(1, 1, 0.2, 0, 0, 0.5, 0.5))
val transformer = DetectionCrop("roi")
val transformed = transformer(data)
```

---
## **Expand**
**Scala:**
```scala
val transformer = Expand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
                    minExpandRatio: Double = 1, maxExpandRatio: Double = 4.0)
```
**Python:**
```python
transformer = Expand(means_r=123, means_g=117, means_b=104,
                                      min_expand_ratio=1.0,
                                      max_expand_ratio=4.0)
```
expand image, fill the blank part with the meanR, meanG, meanB

 * `meansR` means in R channel
 * `meansG` means in G channel
 * `meansB` means in B channel
 * `minExpandRatio` min expand ratio
 * `maxExpandRatio` max expand ratio

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Expand(minExpandRatio = 2, maxExpandRatio = 2)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = Expand(min_expand_ratio = 2.0, max_expand_ratio = 2.0)
transformed = transformer(data)
```

---
## **Filler**
**Scala:**
```scala
val transformer = Filler(startX: Float, startY: Float, endX: Float, endY: Float, value: Int = 255)
```
**Python:**
```python
transformer = Filler(start_x, start_y, end_x, end_y, value = 255)
```
Fill part of image with certain pixel value

 * `startX` start x ratio
 * `startY` start y ratio
 * `endX` end x ratio
 * `endY` end y ratio
 * `value` filling value

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = Filler(0, 0, 1, 0.5f, 255)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = Filler(0.0, 0.0, 1.0, 0.5, 255)
transformed = transformer(data)
```

---
## **HFlip**
**Scala:**
```scala
val transformer = HFlip()
```
**Python:**
```python
transformer = HFlip()
```
Flip the image horizontally

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = HFlip()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = HFlip()
transformed = transformer(data)
```

---
## **RandomTransformer**
**Scala:**
```scala
val transformer = RandomTransformer(transformer: FeatureTransformer, maxProb: Double)
```
**Python:**
```python
transformer = RandomTransformer(transformer, maxProb)
```
It is a wrapper for transformers to control the transform probability
 * `transformer` transformer to apply randomness
 * `maxProb` max prob

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = RandomTransformer(HFlip(), 0.5)
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = RandomTransformer(HFlip(), 0.5)
transformed = transformer(data)
```

---
## **BytesToMat**
**Scala:**
```scala
val transformer = BytesToMat(byteKey: String = ImageFeature.bytes)
```
**Python:**
```python
transformer = BytesToMat(byte_key="bytes")
```
Transform byte array(original image file in byte) to OpenCVMat
* `byteKey`: key that maps byte array

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.BytesToMat
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = BytesToMat()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = BytesToMat()
transformed = transformer(data)
```

---
## **MatToFloats**
**Scala:**
```scala
val transformer = MatToFloats(validHeight: Int, validWidth: Int, validChannels: Int,
                    outKey: String = ImageFeature.floats, shareBuffer: Boolean = true)
```
**Python:**
```python
transformer = MatToFloats(valid_height=300, valid_width=300, valid_channel=300,
                                          out_key = "floats", share_buffer=True)
```
Transform OpenCVMat to float array, note that in this transformer, the mat is released.
 * `validHeight` valid height in case the mat is invalid
 * `validWidth` valid width in case the mat is invalid
 * `validChannels` valid channel in case the mat is invalid
 * `outKey` key to store float array
 * `shareBuffer` share buffer of output

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.MatToFloats
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = MatToFloats()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = MatToFloats()
transformed = transformer(data)
```

---
## **MatToTensor**
**Scala:**
```scala
val transformer = MatToTensor(toRGB: Boolean = false,
                               tensorKey: String = ImageFeature.imageTensor)
```
**Python:**
```python
transformer = MatToTensor(to_rgb=False, tensor_key="imageTensor")
```
Transform opencv mat to tensor, note that in this transformer, the mat is released.
 * `toRGB` BGR to RGB (default is BGR)
 * `tensorKey` key to store transformed tensor

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.MatToTensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = MatToTensor[Float]()
val transformed = transformer(data)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = MatToTensor()
transformed = transformer(data)
```

---
## **ImageFrameToSample**
**Scala:**
```scala
val transformer = ImageFrameToSample(inputKeys: Array[String] = Array(ImageFeature.imageTensor),
                               targetKeys: Array[String] = null,
                               sampleKey: String = ImageFeature.sample)
```
**Python:**
```python
transformer = ImageFrameToSample(input_keys=["imageTensor"], target_keys=None,
                                           sample_key="sample")
```
Transforms tensors that map inputKeys and targetKeys to sample,
note that in this transformer, the mat has been released.
  * `inputKeys` keys that maps inputs (each input should be a tensor)
  * `targetKeys` keys that maps targets (each target should be a tensor)
  * `sampleKey` key to store sample

Note that you may need to chain `MatToTensor` before `ImageFrameToSample`,
since `ImageFrameToSample` requires all inputkeys map Tensor type

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image._

val data = ImageFrame.read("/tmp/test.jpg")
val transformer = MatToTensor[Float]()
val toSample = ImageFrameToSample[Float]()
val transformed = transformer(data)
toSample(transformed)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
data = ImageFrame.read("/tmp/test.jpg")
transformer = MatToTensor()
to_sample = ImageFrameToSample()
transformed = transformer(data)
to_sample(transformed)
```