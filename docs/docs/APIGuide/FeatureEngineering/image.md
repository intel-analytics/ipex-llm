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
def read(path: String, sc: SparkContext = null, minPartitions: Int = 1, resizeH: Int = -1, resizeW: Int = -1, imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED, withLabel: Boolean = false, oneBasedLabel: Boolean = true): ImageSet
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* minPartitions: A suggestion value of the minimal splitting number for input data.
* resizeH: height after resize, by default is -1 which will not resize the image
* resizeW: width after resize, by default is -1 which will not resize the image
* imageCodec: specifying the color type of a loaded image, same as in OpenCV.imread. By default is `Imgcodecs.CV_LOAD_IMAGE_UNCHANGED`.
* withLabel: whether to treat folders in the path as image classification labels and read the labels into ImageSet.
* oneBasedLabel: whether the labels start from 1. If true, the labels starts from 1, else the labels start from 0.
   
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
read(path, sc=None, min_partitions=1, resize_height=-1, resize_width=-1, image_codec=-1, with_label=False, one_based_label=True, bigdl_type="float")
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* min_partitions: A suggestion value of the minimal splitting number for input data.
* resize_height height after resize, by default is -1 which will not resize the image
* resize_width width after resize, by default is -1 which will not resize the image
* image_codec: specifying the color type of a loaded image, same as in OpenCV.imread. By default is -1(`Imgcodecs.CV_LOAD_IMAGE_UNCHANGED`).
* with_label: whether to treat folders in the path as image classification labels and read the labels into ImageSet.
* one_based_label: whether the labels start from 1. By default it is true, else the labels start from 0.


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
## 3D Image Support

### Create ImageSet for 3D Images

For 3D images, you can still use ImageSet as the collection of ImageFeature3D. You can create ImageSet for 3D images in the similar way as for 2D images. Since we do not provide 3D image reader in analytics zoo, before create ImageSet, we suppose you already read 3D images to tensor(scala) or numpy array(python).

Scala example:
```scala
val image = ImageFeature3D(tensor)

// create local imageset for 3D images
val arr = Array[ImageFeature](image)
val localImageSet = ImageSet.array(arr)

// create distributed imageset for 3D images
val rdd = sc.parallelize(Seq[ImageFeature](image))
val imageSet = ImageSet.rdd(rdd)
```

Python example:
```python

# get image numpy array
img_np =

# create local imageset for 3D images
local_imageset = LocalImageSet(image_list=[img_np])

# create distributed imageset for 3D images
rdd = sc.parallelize([img_np])
dist_imageSet = DistributedImageSet(image_rdd=rdd)
```

### 3D Image Transformers
Analytics zoo also provides several image transformers for 3D Images.
The usage is similar as 2D image transformers. After create these transformers, call `transform` with ImageSet to get transformed ImageSet.

Currently we support three kinds of 3D image transformers: Crop, Rotation and Affine Transformation.

#### Crop transformers

##### Crop3D

Scala:
```scala
import com.intel.analytics.zoo.feature.image3d.Crop3D

// create Crop3D transformer
val cropper = Crop3D(start, patchSize)
val outputImageSet = imageset.transform(cropper)
```

Crop a patch from a 3D image from 'start' of patch size. The patch size should be less than the image size.
   * start: start point array(depth, height, width) for cropping
   * patchSize: patch size array(depth, height, width)

Python:
```python
from zoo.feature.image3d.transformation import Crop3D

crop = Crop3D(start, patch_size)
transformed_image = crop(image_set)
```
* start: start point list[]depth, height, width] for cropping
* patch_size: patch size list[]depth, height, width]

##### RandomCrop3D

Scala:
```scala
import com.intel.analytics.zoo.feature.image3d.RandomCrop3D

// create Crop3D transformer
val cropper = RandomCrop3D(cropDepth, cropHeight, cropWidth)
val outputImageSet = imageset.transform(cropper)
```

Crop a random patch from an 3D image with specified patch size. The patch size should be less than the image size.
* cropDepth: depth after crop
* cropHeight: height after crop
* cropWidth: width after crop

Python:
```python
from zoo.feature.image3d.transformation import RandomCrop3D

crop = RandomCrop3D(crop_depth, crop_height, crop_width)
transformed_image = crop(image_set)
```
* crop_depth: depth after crop
* crop_height: height after crop
* crop_width: width after crop

##### CenterCrop3D

Scala:
```scala
import com.intel.analytics.zoo.feature.image3d.CenterCrop3D

// create Crop3D transformer
val cropper = CenterCrop3D(cropDepth, cropHeight, cropWidth)
val outputImageSet = imageset.transform(cropper)
```

Crop a `cropDepth` x `cropWidth` x `cropHeight` patch from center of image. The patch size should be less than the image size.
* cropDepth: depth after crop
* cropHeight: height after crop
* cropWidth: width after crop

Python:
```python
from zoo.feature.image3d.transformation import CenterCrop3D

crop = CenterCrop3D(crop_depth, crop_height, crop_width)
transformed_image = crop(image_set)
```
* crop_depth: depth after crop
* crop_height: height after crop
* crop_width: width after crop

#### Rotation
Scala:
```scala
import com.intel.analytics.zoo.feature.image3d.Rotate3D

// create Crop3D transformer
val rotAngles = Array[Double](yaw, pitch, roll)
val rot = Rotate3D(rotAngles)
val outputImageSet = imageset.transform(rot)
```

Rotate a 3D image with specified angles.
* rotationAngles: the angles for rotation.
   Which are the yaw(a counterclockwise rotation angle about the z-axis),
   pitch(a counterclockwise rotation angle about the y-axis),
   and roll(a counterclockwise rotation angle about the x-axis).

Python:
```python
from zoo.feature.image3d.transformation import Rotate3D

rot = Rotate3D(rotation_angles)
transformed_image = rot(image_set)
```

#### Affine Transformation
Scala:
```scala
import com.intel.analytics.zoo.feature.image3d.AffineTransform3D
import com.intel.analytics.bigdl.tensor.Tensor

// create Crop3D transformer
val matArray = Array[Double](1, 0, 0, 0, 1.5, 1.2, 0, 1.3, 1.4)
val matTensor = Tensor[Double](matArray, Array[Int](3, 3))
val trans = Tensor[Double](3)
trans(1) = 0
trans(2) = 1.8
trans(3) = 1.1
val aff = AffineTransform3D(mat=matTensor, translation = trans, clampMode = "clamp", padVal = 0)
val outputImageSet = imageset.transform(aff)
```
Affine transformer implements affine transformation on a given tensor.
To avoid defects in resampling, the mapping is from destination to source.
dst(z,y,x) = src(f(z),f(y),f(x)) where f: dst -> src

* mat: [Tensor[Double], dim: DxHxW] defines affine transformation from dst to src.
* translation: [Tensor[Double], dim: 3, default: (0,0,0)] defines translation in each axis.
* clampMode: [String, (default: "clamp",'padding')] defines how to handle interpolation off the input image.
* padVal: [Double, default: 0] defines padding value when clampMode="padding". Setting this value when clampMode="clamp" will cause an error.

Python:
```python
from zoo.feature.image3d.transformation import AffineTransform3D

affine = AffineTransform3D(affine_mat, translation, clamp_mode, pad_val)
transformed_image = affine(image_set)
```
* affine_mat: numpy array in 3x3 shape.Define affine transformation from dst to src.
* translation: numpy array in 3 dimension.Default value is np.zero(3). Define translation in each axis.
* clamp_mode: str, default value is "clamp". Define how to handle interpolation off the input image.
* pad_val: float, default is 0.0. Define padding value when clampMode="padding". Setting this value when clampMode="clamp" will cause an error.
