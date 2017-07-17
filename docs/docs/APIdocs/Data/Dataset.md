## DataSet
`DataSet` is a set of data which is used in the model optimization process. You can use `DataSet.array()` and `DataSet.rdd()` function to create a `Dataset`. The `DataSet` can be accessed in a random data sample sequence. In the training process, the data sequence is a looped endless sequence. While in the validation process, the data sequence is a limited length sequence. User can use the `data()` method to get the data sequence. 

Notice: In most case, we recommand using a RDD[Sample] for `Optimizer`. Only when you want to write an application with some advanced optimization, using `DataSet` directly is recommanded.  

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
