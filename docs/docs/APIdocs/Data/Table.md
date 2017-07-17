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
