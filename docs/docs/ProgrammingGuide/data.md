---
## **ImageSet**
`ImageSet` is a collection of `ImageFeature`.
It can be a `DistributedImageSet` for distributed image RDD or
 `LocalImageSet` for local image array.
You can read an `ImageSet` from local/distributed folder,
or you can directly construct a ImageSet from RDD[ImageFeature] or Array[ImageFeature].

**Scala example:**

Create LocalImageSet, assume there is an image file "/tmp/test.jpg"
and an image folder "/tmp/image/"

```scala
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.ImageSet

// create LocalImageSet from an image
val localImageSet = ImageSet.read("/tmp/test.jpg")

// create LocalImageSet from an image folder
val localImageSet2 = ImageSet.read("/tmp/image/")

// create LocalImageSet from array of ImageFeature
val array = Array[ImageFeature]()
val localImageSet3 = ImageSet.array(array)
```
Create DistributedImageSet, assume there is an image file "/tmp/test.jpg"
and an image folder

```scala
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.ImageSet
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

val conf = new SparkConf().setAppName("ImageSpec").setMaster("local[2]")
val sc = NNContext.initNNContext(conf)
val sqlContext = new SQLContext(sc)

// create DistributedImageSet from an image
val distributedImageSet = ImageSet.read("/tmp/test.jpg", sc, 2)

// create DistributedImageSet from an image folder
val distributedImageSet2 = ImageSet.read("/tmp/image/", sc, 2)

// create DistributedImageSet from rdd of ImageFeature
val array = Array[ImageFeature]()
val rdd = sc.parallelize(array)
val distributedImageSet3 = ImageSet.rdd(rdd)
```

**Python example:**

Create LocalImageSet

```python
from bigdl.util.common import *
from zoo.feature.image.imageset import *

# create LocalImageSet from an image
local_image_set = ImageSet.read("/tmp/test.jpg")

# create LocalImageSet from an image folder
local_image_set2 = ImageSet.read("/tmp/image/")

# create LocalImageSet from list of images
image = cv2.imread("/tmp/test.jpg")
local_image_set3 = LocalImageSet([image])
```

Create DistributedImageSet

```python
from zoo.common.nncontext import *
from zoo.feature.image.imageset import *

sc = init_nncontext(init_spark_conf().setMaster("local[2]").setAppName("test image"))
# create DistributedImageSet from an image
distributed_image_set = ImageSet.read("/tmp/test.jpg", sc, 2)

# create DistributedImageSet from an image folder
distributed_image_set = ImageSet.read("/tmp/image/", sc, 2)

# create DistributedImageSet from image rdd
image = cv2.imread("/tmp/test.jpg")
image_rdd = sc.parallelize([image], 2)
distributed_image_set = DistributedImageSet(image_rdd)
```