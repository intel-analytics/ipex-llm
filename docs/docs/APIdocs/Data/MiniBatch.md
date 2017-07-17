## MiniBatch
`MiniBatch` is a data structure to feed input/target to model in `Optimizer`. It provide `getInput()` and `getTarget()` function to get the input and target in this `MiniBatch`.

In almost all the cases, BigDL's default `MiniBatch` class can fit user's requirement. User can implement their own `MiniBatch` class by extends [MiniBatch](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/dataset/MiniBatch.scala).

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

If you `Sample`s are not the same size, you can use `PaddingParam` to pad the `Sample`s to the same size.
```scala
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

val sample1 = Sample(Tensor(T(1f, 2f, 3f, 4f, 5f, 6f)).resize(2, 3), 1f)
val sample2 = Sample(Tensor(T(7f, 8f, 9f)).resize(1, 3), 2f)
val sample3 = Sample(Tensor(T(10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f)).resize(3, 3), 3f)
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

