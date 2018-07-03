Analytics Zoo provides a series of Image APIs for end-to-end image processing pipeline, including image loading, pre-processing, inference/training and some utilities on different formats.

## Load Image
Analytics Zoo provides APIs to read image to different formats:

### Load to Data Frame
Scala:
```scala
package com.intel.analytics.zoo.pipeline.nnframes

object NNImageReader {
  def readImages(path: String, sc: SparkContext, minPartitions: Int = 1, resizeH: Int = -1, resizeW: Int = -1): DataFrame
}
```

Read the directory of images from the local or remote source, return DataFrame with a single column "image" of images.

* path: Directory to the input data files, the path can be comma separated paths as the list of inputs. Wildcards path are supported similarly to sc.binaryFiles(path).
* sc: SparkContext to be used.
* minPartitions: Number of the DataFrame partitions, if omitted uses defaultParallelism instead
* resizeH: height after resize, by default is -1 which will not resize the image
* resizeW: width after resize, by default is -1 which will not resize the image
  
Python:
```python
class zoo.pipeline.nnframes.NNImageReader
    static readImages(path, sc=None, minPartitions=1, resizeH=-1, resizeW=-1, bigdl_type="float")
```
### ImageSet
`ImageSet` is a collection of `ImageFeature`. It can be a `DistributedImageSet` for distributed image RDD or
 `LocalImageSet` for local image array.
You can read image data as `ImageSet` from local/distributed image path, or you can directly construct a ImageSet from RDD[ImageFeature] or Array[ImageFeature].

**Scala APIs:**

```scala
object com.intel.analytics.zoo.feature.image.ImageSet
```

```
def array(data: Array[ImageFeature]): LocalImageSet
```
Create LocalImageSet from array of ImeageFeature
  
* data: array of ImageFeature

```
def rdd(data: RDD[ImageFeature]): DistributedImageSet
```
Create DistributedImageSet from rdd of ImageFeature

* data: array of ImageFeature
```
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1, resizeH: Int = -1, resizeW: Int = -1): ImageSet
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* minPartitions: A suggestion value of the minimal splitting number for input data.
* resizeH: height after resize, by default is -1 which will not resize the image
* resizeW: width after resize, by default is -1 which will not resize the image
   
Example:
```
// create LocalImageSet from an image folder
val localImageSet = ImageSet.read("/tmp/image/")

// create DistributedImageSet from an image folder
val distributedImageSet2 = ImageSet.read("/tmp/image/", sc, 2)
```

**Python APIs:**

```
class zoo.feature.image.ImageSet
```
```
read(path, sc=None, min_partitions=1, resize_height=-1, resize_width=-1, bigdl_type="float")
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* min_partitions: A suggestion value of the minimal splitting number for input data.
* resize_height height after resize, by default is -1 which will not resize the image
* resize_width width after resize, by default is -1 which will not resize the image

Python example:
```python
# create LocalImageSet from an image folder
local_image_set2 = ImageSet.read("/tmp/image/")

# create DistributedImageSet from an image folder
distributed_image_set = ImageSet.read("/tmp/image/", sc, 2)
```

## Image Transformer
Analytics Zoo provides many pre-defined image processing transformers built on top of OpenCV. After create these transformers, call `transform` with ImageSet to get transformed ImageSet. Or pass the transformer to NNEstimator/NNClassifier to preprocess before training. 

**Scala APIs:**
```
package com.intel.analytics.zoo.feature.image

object ImageBrightness

def apply(deltaLow: Double, deltaHigh: Double): ImageBrightness
```
Adjust the image brightness.

* deltaLow: low bound of brightness parameter
* deltaHigh: high bound of brightness parameter

Example:
```
val transformer = ImageBrightness(0.0, 32.0)
val transformed = imageSet.transform(transformer)
```

**Python APIs:**
```
class zoo.feature.image.imagePreprocessing.ImageBrightness

def __init__(delta_low, delta_high, bigdl_type="float")
```
Adjust the image brightness.

* delta_low: low bound of brightness parameter
* delta_high: high bound of brightness parameter

Example:
```
transformer = ImageBrightness(0.0, 32.0)
transformed = imageSet.transform(transformer)
```

**Scala APIs:**
```
package com.intel.analytics.zoo.feature.image

object ImageBytesToMat

def apply(byteKey: String = ImageFeature.bytes,
          imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): ImageBytesToMat
```
Transform byte array(original image file in byte) to OpenCVMat

* byteKey: key that maps byte array. Default value is ImageFeature.bytes
* imageCodec: specifying the color type of a loaded image, same as in OpenCV.imread.
              1. CV_LOAD_IMAGE_ANYDEPTH - If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
              2. CV_LOAD_IMAGE_COLOR - If set, always convert image to the color one
              3. CV_LOAD_IMAGE_GRAYSCALE - If set, always convert image to the grayscale one
              4. >0 Return a 3-channel color image.
              Note The alpha channel is stripped from the output image. Use negative value if you need the alpha channel.
              5. =0 Return a grayscale image.
              6. <0 Return the loaded image as is (with alpha channel).
              Default value is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED.

Example:
```
val imageSet = ImageSet.read(path, sc)
imageSet -> ImageBytesToMat()
```
