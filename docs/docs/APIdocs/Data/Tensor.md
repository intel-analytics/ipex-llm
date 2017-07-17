## Tensor

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
scala> val a = Tensor(T(1f, 2f, 3f, 4f, 5f, 6f)).resize(2, 3)
a: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0	2.0	3.0
4.0	5.0	6.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]

scala> val b = Tensor(T(6f, 5f, 4f, 3f, 2f, 1f)).resize(2, 3)
b: com.intel.analytics.bigdl.tensor.Tensor[Float] =
6.0	5.0	4.0
3.0	2.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
```
`+` `-` `*` `/` can be applied to tensor. When the second parameter is a constant value, `+` `-` `*` `*` is element-wise operation. But when the second parameter is a tensor, `+` `-` `/` is element-wise operation to the tensor too, but `*` is a matrix multipy on two 2D tensors. 
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
For more API, please visit [bigdl scala doc](https://bigdl-project.github.io/latest/#APIdocs/scaladoc/)

