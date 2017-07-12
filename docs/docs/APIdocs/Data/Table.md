## **Table**

Modeled after the [Table](https://github.com/torch/nn/blob/master/doc/table.md) class in [Torch](http://torch.ch/), the  ```Table``` class (defined in package ```com.intel.analytics.bigdl.utils```) is widely used in BigDL (e.g., a            ```Table``` of ```Tensor``` can be used as the input or output of neural networks). In essence, a ```Table``` can be     considered as a key-value map, and there is also a syntax sugar to create a ```Table``` using ```T()``` in BigDL.

```scala

scala> import com.intel.analytics.bigdl.utils.T
 import com.intel.analytics.bigdl.utils.T

 scala> T(Tensor[Float](2,2), Tensor[Float](2,2))
 res2: com.intel.analytics.bigdl.utils.Table =
  {
         2: 0.0  0.0
            0.0  0.0
            [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
         1: 0.0  0.0
            0.0  0.0
            [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
  }

```
