A FeatureSet can be used to represent an input pipeline as a collection of elements which is used in the model optimization process. You can use FeatureSet to switch the memory type between `DRAM`, `DISK_AND_DRAM(numSlice)` 
and `PMEM` in consideration of the hardware optimization.
* `DRAM` is the default mode which would cached the training data in main memory.
* `DISK_AND_DRAM(numSlice)` mode will cache training data in disk, and only hold 1/numSlice of data in main memory. After going through the 1/numSlice, we will release the current cache, and load another 1/numSlice into memory.
* `PMEM` mode would try to cache the training data in persistent memory rather than main memory. You should install the Intel Optane DC Persistent Memory hardware to AD mode before switching to this option. 
 


Scala example:

In scala, user can create a DRAM or DISK_AND_DRAM FeatureSet from any RDD. 
But PMEM FeatureSet only support RDD of ByteRecord, Sample or ImageFeature.

```scala
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.pmem.{DRAM, DISK_AND_DRAM, PMEM}
// DRAM featureSet
val dramSet = FeatureSet.rdd(rawRDD, memoryType = DRAM)
// DISK_AND_DRAM featureSet
val diskSet = FeatureSet.rdd(rawRDD, memoryType = DISK_AND_DRAM(10))
// PMEM featureSet
val pmemSet = FeatureSet.rdd(rawRDD, memoryType = PMEM)
// Add your transformer: featureSet -> feature transformer -> batch and sample transformer
val dataSet = ...
```

Take a look at [InceptionV1 example scala version](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception) for more details.

Python example:
In python, user can create FeatureSet from RDD, ImageSet, and ImageFrame.
```python
from zoo.feature.common import FeatureSet
dramSet = FeatureSet.rdd(rawRDD, memory_type = "DRAM")
# DISK_AND_DRAM featureSet
diskSet = FeatureSet.rdd(rawRDD, memoryType = "10")
#PMEM featureSet
pmemSet = FeatureSet.rdd(rawRDD, memoryType = "PMEM")
#Add your transformer: featureSet -> feature transformer -> batch and sample transformer
dataSet = ...
```

Take a look at [InceptionV1 example python version](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/examples/inception/inception.py) for more details.
