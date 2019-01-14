A FeatureSet can be used to represent an input pipeline as a collection of elements which is used in the model optimization process. You can use FeatureSet to switch the memory type between `DRAM` 
and `PMEM` in consideration of the hardware optimization.
* `DRAM` is the default mode which would cached the training data in main memory.
* `PMEM` mode would try to cache the training data in AEP rather than main memory. You should install the AEP hardware and [memkind library](https://github.com/memkind/memkind) before switching
 to this option. 
 
* The FeatureSet can be accessed in a random data sample sequence. In the training process, the data sequence is a looped endless sequence. While in the validation process, the data sequence is a limited length sequence. User can use the data() method to get the data sequence.
* You can use FeatureSet.rdd() function to create a FeatureSet.

Scala example:

```scala
   import com.intel.analytics.zoo.feature.FeatureSet
   val featureSet = FeatureSet.rdd(rawRDD, memoryType = DRAM)
   // featureSet -> feature transformer -> batch and sample transformer
   model.fit(featureSet)
```
Take a look at [InceptionV1 example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception) for more details.