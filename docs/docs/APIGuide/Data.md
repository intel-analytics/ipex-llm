## **Tensor**

Modeled after the [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md) class in [Torch](http://torch.ch/  ), the ```Tensor``` [package](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) (written in Scala and leveraging [Intel MKL](https://software.intel.com/en-us/intel-mkl)) in BigDL provides numeric computing support for the deep learning applications (e.g., the input, output, weight, bias and   gradient of the neural networks).

A ```Tensor``` is essentially a multi-dimensional array of numeric types (```Float``` or ```Double```), you can import the numeric implicit objects(`com.intel.analytics.bigdl.numeric.NumericFloat` or `com.intel.analytics.bigdl.numeric.NumericDouble`), to specify the numeric type you want.


**Scala example:**

You may check it out in the interactive Scala shell (by typing ```scala -cp bigdl_SPARKVERSION-BIGDLVERSION-SNAPSHOT-jar-with-dependencies.jar```), for instance:

```scala
 scala> import com.intel.analytics.bigdl.tensor.Tensor
 import com.intel.analytics.bigdl.tensor.Tensor
 
 scala> import com.intel.analytics.bigdl.numeric.NumericFloat
 import com.intel.analytics.bigdl.numeric.NumericFloat
 
 scala> import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.T

 scala> val tensor = Tensor(2, 3)
 tensor: com.intel.analytics.bigdl.tensor.Tensor =
 0.0     0.0     0.0
 0.0     0.0     0.0
 [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Tensor can be created with existing data.
```scala
scala> val a = Tensor(T(
      T(1f, 2f, 3f),
      T(4f, 5f, 6f)))
a: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0	2.0	3.0
4.0	5.0	6.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> val b = Tensor(T(
      T(6f, 5f, 4f),
      T(3f, 2f, 1f)))
b: com.intel.analytics.bigdl.tensor.Tensor[Float] =
6.0	5.0	4.0
3.0	2.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
```
`+` `-` `*` `/` can be applied to tensor. When the second parameter is a constant value, `+` `-` `*` `/` is element-wise operation. But when the second parameter is a tensor, `+` `-` `/` is element-wise operation to the tensor too, but `*` is a matrix multiply on two 2D tensors. 
```scala
scala> a + 1
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
2.0	3.0	4.0
5.0	6.0	7.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> a + b
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
7.0	7.0	7.0
7.0	7.0	7.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

scala> a - b
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-5.0	-3.0	-1.0
1.0	3.0	5.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]

scala> a * b.t
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
28.0	10.0
73.0	28.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]

scala> a / b
res: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.16666667	0.4	0.75
1.3333334	2.5	6.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
For more API, navigate to *API Guide/Full API docs* on side bar.

---
## **SparseTensor**
To describe an SparseTensor, we need indices, values, and shape:  
indices means the indices of non-zero elements, they should be zero-based and ascending;
values means the values of the non-zero elements;
shape means the dense shape of this SparseTensor.

For example, an 2D 3x4 DenseTensor:
```
1, 0, 0, 4
0, 2, 0, 0
0, 0, 3, 0
```
It's sparse representation should be 
```
indices(0) = Array(0, 0, 1, 2)
indices(1) = Array(0, 3, 1, 2)
values     = Array(1, 4, 2, 3)
shape      = Array(3, 4)
```
This 2D SparseTensor representation is similar to [zero-based coordinate matrix storage format](https://software.intel.com/en-us/mkl-developer-reference-fortran-sparse-blas-coordinate-matrix-storage-format).

**Scala example:**
```
scala> import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Tensor

scala> import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.numeric.NumericFloat

scala> val indices = Array(Array(0, 0, 1, 2), Array(0, 3, 1, 2))
indices: Array[Array[Int]] = Array(Array(0, 0, 1, 2), Array(0, 3, 1, 2))

scala> val values = Array(1, 4, 2, 3)
values: Array[Int] = Array(1, 4, 2, 3)

scala> val shape = Array(3, 4)
shape: Array[Int] = Array(3, 4)

scala> val sparseTensor = Tensor.sparse(indices, values, shape)
sparseTensor: com.intel.analytics.bigdl.tensor.Tensor[Int] =
(0, 0) : 1
(0, 3) : 4
(1, 1) : 2
(2, 2) : 3
[com.intel.analytics.bigdl.tensor.SparseTensor of size 3x4]

scala> val denseTensor = Tensor.dense(sparseTensor)
denseTensor: com.intel.analytics.bigdl.tensor.Tensor[Int] =
1	0	0	4
0	2	0	0
0	0	3	0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
```

---
## **Table**

Modeled after the [Table](https://github.com/torch/nn/blob/master/doc/table.md) class in [Torch](http://torch.ch/), the ```Table``` class (defined in package ```com.intel.analytics.bigdl.utils```) is widely used in BigDL (e.g., a ```Table``` of ```Tensor``` can be used as the input or output of neural networks). In essence, a ```Table``` can be considered as a key-value map, and there is also a syntax sugar to create a ```Table``` using ```T()``` in BigDL.

**Scala example:**
```scala
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
println(T(Tensor(2,2).fill(1), Tensor(2,2).fill(2)))
```
Output is
```scala
 {
	2: 2.0	2.0	
	   2.0	2.0	
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
	1: 1.0	1.0	
	   1.0	1.0	
	   [com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2]
 }
```
---
## **Sample**

A `Sample` represents one record of your data set, which is comprised of `feature` and `label`.

- `feature` is one tensor or a few tensors
- `label` is also one tensor or a few tensors, and it may be empty in testing or unsupervised learning.

For example, one image and its category in image classification, one word in word2vec and one sentence and its label in RNN language model are all `Sample`.

Every `Sample` is actually a set of tensors, and they will be transformed to the input/output of the model. For example, in the case of image classification, a `Sample` has two tensors. One is a 3D tensor representing an image; another is a 1-element tensor representing its category. For the 1-element label, you also can use a `T` instead of tensor.

**Scala example:**

- The case where feature is one tensor with a 1-element label.

```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val image = Tensor(3, 32, 32).rand
val label = 1f
val sample = Sample(image, label)
```

- The case where feature is a few tensors and label is also a few tensors.

```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val features = Array(Tensor(2, 2).rand, Tensor(2, 2).rand)
val labels = Array(Tensor(1).fill(1), Tensor(1).fill(-1))
val sample = Sample(features, labels)
```

**Python example:**

__Note__: Please always use `Sample.from_ndarray` to construct a `Sample` in Python.

- The case where feature is one tensor with a 1-element label.

After constructing a `Sample` in this case, you can use `Sample.feature` and `Sample.label` to retrieve its feature and label, each as a tensor, respectively.

```python
from bigdl.util.common import Sample
import numpy as np

image = np.random.rand(3, 32, 32)
label = np.array(1)
sample = Sample.from_ndarray(image, label)

# Retrieve feature and label from a Sample
sample.feature
sample.label
```

- The case where feature is a few tensors and label is also a few tensors.

After constructing a `Sample` in this case, you can use `Sample.features` and `Sample.labels` to retrieve its features and labels, each as a list of tensors, respectively.

```python
from bigdl.util.common import Sample
import numpy as np

features = [np.random.rand(3, 8, 16), np.random.rand(3, 8, 16)]
labels = [np.array(1), np.array(-1)]
sample = Sample.from_ndarray(features, labels)

# Retrieve features and labels from a Sample
sample.features
sample.labels
```
Note that essentially `Sample.label` is equivalent to `Sample.labels[0]`. You can choose to use the former if label is only one tensor and use the latter if label is a list of tensors. Similarly, `Sample.feature` is equivalent to `Sample.features[0]`.

---
## **MiniBatch**

`MiniBatch` is a data structure to feed input/target to model in `Optimizer`. It provide `getInput()` and `getTarget()` function to get the input and target in this `MiniBatch`.

In almost all the cases, BigDL's default `MiniBatch` class can fit user's requirement. Just create your `RDD[Sample]` and pass it to `Optimizer`. If `MiniBatch` can't meet your requirement, you can implement your own `MiniBatch` class by extends [MiniBatch](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset/MiniBatch.scala).

`MiniBatch` can be created by `MiniBatch(nInputs: Int, nOutputs: Int)`, `nInputs` means number of inputs, `nOutputs` means number of outputs. And you can use `set(samples: Seq[Sample[T])` to fill the content in this MiniBatch. If you `Sample`s are not the same size, you can use `PaddingParam` to pad the `Sample`s to the same size.

**Scala example:**
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val samples  = Array.tabulate(5)(i => Sample(Tensor(1, 3, 3).fill(i), i + 1f))
val miniBatch = MiniBatch(1, 1).set(samples)
println(miniBatch.getInput())
println(miniBatch.getTarget())
```
Output is
```scala
(1,1,.,.) =
0.0	0.0	0.0	
0.0	0.0	0.0	
0.0	0.0	0.0	

(2,1,.,.) =
1.0	1.0	1.0	
1.0	1.0	1.0	
1.0	1.0	1.0	

(3,1,.,.) =
2.0	2.0	2.0	
2.0	2.0	2.0	
2.0	2.0	2.0	

(4,1,.,.) =
3.0	3.0	3.0	
3.0	3.0	3.0	
3.0	3.0	3.0	

(5,1,.,.) =
4.0	4.0	4.0	
4.0	4.0	4.0	
4.0	4.0	4.0	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x1x3x3]
1.0	
2.0	
3.0	
4.0	
5.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x1]
```

If your `Sample`s are not the same size, you can use `PaddingParam` to pad the `Sample`s to the same size.
```scala
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val sample1 = Sample(Tensor.range(1, 6, 1).resize(2, 3), 1f)
val sample2 = Sample(Tensor.range(7, 9, 1).resize(1, 3), 2f)
val sample3 = Sample(Tensor.range(10, 18, 1).resize(3, 3), 3f)
val samples = Array(sample1, sample2, sample3)
val featurePadding = PaddingParam(Some(Array(Tensor(T(-1f, -2f, -3f)))), FixedLength(Array(4)))
val labelPadding = PaddingParam[Float](None, FixedLength(Array(4)))

val miniBatch = MiniBatch(1, 1, Some(featurePadding), Some(labelPadding)).set(samples)
println(miniBatch.getInput())
println(miniBatch.getTarget())
```
Output is 
```
(1,.,.) =
1.0	2.0	3.0	
4.0	5.0	6.0	
-1.0	-2.0	-3.0	
-1.0	-2.0	-3.0	

(2,.,.) =
7.0	8.0	9.0	
-1.0	-2.0	-3.0	
-1.0	-2.0	-3.0	
-1.0	-2.0	-3.0	

(3,.,.) =
10.0	11.0	12.0	
13.0	14.0	15.0	
16.0	17.0	18.0	
-1.0	-2.0	-3.0	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x3]


1.0	0.0	0.0	0.0	
2.0	0.0	0.0	0.0	
3.0	0.0	0.0	0.0	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
```




## **DataSet**
`DataSet` is a set of data which is used in the model optimization process. You can use `DataSet.array()` and `DataSet.rdd()` function to create a `Dataset`. The `DataSet` can be accessed in a random data sample sequence. In the training process, the data sequence is a looped endless sequence. While in the validation process, the data sequence is a limited length sequence. User can use the `data()` method to get the data sequence. 

Notice: In most case, we recommend using a RDD[Sample] for `Optimizer`. Only when you want to write an application with some advanced optimization, using `DataSet` directly is recommended.  

**Scala example:**
```scala
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dataset.DataSet

val tensors  = Array.tabulate(5)(i => Tensor(1, 3, 3).fill(i))
val dataset = DataSet.array(tensors) // Local model, just for testing and example.
dataset.shuffle()
val iter = dataset.data(false)
while (iter.hasNext) {
  val d = iter.next()
  println(d)
}
```
Output may be
```scala
(1,.,.) =
4.0	4.0	4.0	
4.0	4.0	4.0	
4.0	4.0	4.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
(1,.,.) =
0.0	0.0	0.0	
0.0	0.0	0.0	
0.0	0.0	0.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
(1,.,.) =
2.0	2.0	2.0	
2.0	2.0	2.0	
2.0	2.0	2.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
(1,.,.) =
1.0	1.0	1.0	
1.0	1.0	1.0	
1.0	1.0	1.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
(1,.,.) =
3.0	3.0	3.0	
3.0	3.0	3.0	
3.0	3.0	3.0	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x3x3]
```
---
## **OpenCVMat**
OpenCVMat is a Serializable wrapper of org.opencv.core.Mat.

It can be created by
* `read`: read local image path as opencv mat
* `fromImageBytes`: convert image file in bytes to opencv mat
* `fromFloats`: convert float array(pixels) to OpenCV mat
* `fromTensor`: convert float tensor to OpenCV mat

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.commons.io.FileUtils
import java.io.File

// read local image path as OpenCVMat
val mat = OpenCVMat.read("/tmp/test.jpg")

// convert image file in bytes to OpenCVMat
val bytes = FileUtils.readFileToByteArray(new File(path))
val mat2 = OpenCVMat.fromImageBytes(bytes)

// Convert float array(pixels) to OpenCVMat
val mat3 = OpenCVMat.fromFloats(floatPixels, height=300, width=300)

// Convert tensor to OpenCVMat
val mat4 = OpenCVMat.fromTensor(tensor, format = "HWC")
```

---
## **ImageFeature**
`ImageFeature` is a representation of one image.
It can include various status of an image, by using key-value store.
The key is string that identifies the corresponding value.
Some predefined keys are listed as follows:
* uri: uri that identifies image
* mat: image in OpenCVMat
* bytes: image file in bytes
* floats: image pixels in float array
* size: current image size (height, width, channel)
* originalSize: original image size (height, width, channel)
* label: image label
* predict: image prediction result
* boundingBox: store boundingBox of current image,
it may be used in crop/expand that may change the size of image
* sample: image (and label if available) stored as Sample
* imageTensor: image pixels in Tensor

Besides the above keys, you can also define your key and store information needed
in the prediction pipeline.

**Scala example:**
```scala
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import org.apache.commons.io.FileUtils
import java.io.File

val file = new File("/tmp/test.jpg")
val imageFeature = ImageFeature(FileUtils.readFileToByteArray(file), uri = file.getAbsolutePath)
println(imageFeature.keys())
```

output is

```
Set(uri, bytes)
```

**Python example:**
```python
from bigdl.transform.vision.image import *
image = cv2.imread("/tmp/test.jpg")
image_feature = ImageFeature(image)
print image_feature.keys()
```

output is
```
creating: createImageFeature
[u'originalSize', u'mat', u'bytes']
```

---
## **ImageFrame**
`ImageFrame` is a collection of `ImageFeature`.
It can be a `DistributedImageFrame` for distributed image RDD or
 `LocalImageFrame` for local image array.
You can read an `ImageFrame` from local/distributed folder/parquet file,
or you can directly construct a ImageFrame from RDD[ImageFeature] or Array[ImageFeature].

**Scala example:**

Create LocalImageFrame, assume there is an image file "/tmp/test.jpg"
and an image folder "/tmp/image/"

```scala
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature

// create LocalImageFrame from an image
val localImageFrame = ImageFrame.read("/tmp/test.jpg")

// create LocalImageFrame from an image folder
val localImageFrame2 = ImageFrame.read("/tmp/image/")

// create LocalImageFrame from array of ImageFeature
val array = Array[ImageFeature]()
val localImageFrame3 = ImageFrame.array(array)
```
Create DistributedImageFrame, assume there is an image file "/tmp/test.jpg"
and an image folder

```scala
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

val conf = Engine.createSparkConf().setAppName("ImageSpec").setMaster("local[2]")
val sc = new SparkContext(conf)
val sqlContext = new SQLContext(sc)

// create DistributedImageFrame from an image
val distributedImageFrame = ImageFrame.read("/tmp/test.jpg", sc, 2)

// create DistributedImageFrame from an image folder
val distributedImageFrame2 = ImageFrame.read("/tmp/image/", sc, 2)

// create DistributedImageFrame from rdd of ImageFeature
val array = Array[ImageFeature]()
val rdd = sc.parallelize(array)
val distributedImageFrame3 = ImageFrame.rdd(rdd)

// create DistributedImageFrame from Parquet
val distributedImageFrame4 = ImageFrame.readParquet(dir, sqlContext)
```

**Python example:**

Create LocalImageFrame

```python
from bigdl.util.common import *
from bigdl.transform.vision.image import *

# create LocalImageFrame from an image
local_image_frame = ImageFrame.read("/tmp/test.jpg")

# create LocalImageFrame from an image folder
local_image_frame2 = ImageFrame.read("/tmp/image/")

# create LocalImageFrame from list of images
image = cv2.imread("/tmp/test.jpg")
local_image_frame3 = LocalImageFrame([image])
```

Create DistributedImageFrame

```python
from bigdl.util.common import *
from bigdl.transform.vision.image import *

sparkConf = create_spark_conf().setMaster("local[2]").setAppName("test image")
sc = get_spark_context(sparkConf)
init_engine()

# create DistributedImageFrame from an image
distributed_image_frame = ImageFrame.read("/tmp/test.jpg", sc, 2)

# create DistributedImageFrame from an image folder
distributed_image_frame = ImageFrame.read("/tmp/image/", sc, 2)

# create DistributedImageFrame from image rdd
image = cv2.imread("/tmp/test.jpg")
image_rdd = sc.parallelize([image], 2)
distributed_image_frame = DistributedImageFrame(image_rdd)
```