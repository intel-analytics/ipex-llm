## MiniBatch
MiniBatch` is a data structure to feed input/target to model in `Optimizer`. It provide `getInput()` and `getTarget` function to get the input and target in this MiniBatch.

`MiniBatch` can be created by `MiniBatch(nInputs: Int, nOutputs: Int)`, `nInputs` means number of inputs, `nOutputs` means number of outputs. And you can use `set(samples: Seq[Sample[T])` to fill the content in this MiniBatch.

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
