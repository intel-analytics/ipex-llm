A FeatureSet can be used to represent an input pipeline as a collection of elements which is used in the model optimization process. You can use FeatureSet to switch between `DRAM` and `PMEM` data mode. `DRAM` is the default mode which would cached the training data in main memory. For `PMEM` mode, you should install the AEP hardware first, then it would try to cache the training data in AEP rather than main memory.
* You can use FeatureSet.rdd() function to create a FeatureSet.
* FeatureSet can support multiple memory modes in consideration of the hardware optimization. i.e `DRAM` and `PMEM`
* The FeatureSet can be accessed in a random data sample sequence. In the training process, the data sequence is a looped endless sequence. While in the validation process, the data sequence is a limited length sequence. User can use the data() method to get the data sequence.
* In most case, we recommend using a RDD[Sample] for Optimizer. Only when you want to write an application with some advanced optimization, using FeatureSet directly is recommended.

Scala example:

```scala
   import com.intel.analytics.zoo.feature.FeatureSet
   val featureSet = FeatureSet.rdd(rawRDD, memoryType = DRAM)
   // featureSet -> feature transformer -> batch and sample transformer
   model.fit(featureSet)
```
Take a look at [InceptionV1 example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception) for more details.