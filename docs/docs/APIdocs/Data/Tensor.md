## **Tensor**

Modeled after the [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md) class in [Torch](http://torch.ch/  ), the ```Tensor``` [package](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor) (written in Scala and leveraging [Intel MKL](https://software.intel.com/en-us/intel-mkl)) in BigDL provides numeric computing support for the deep learning applications (e.g., the input, output, weight, bias and   gradient of the neural networks).

A ```Tensor``` is essentially a multi-dimensional array of numeric types (```Float``` or ```Double```), you can import the numeric implicit objects(`com.intel.analytics.bigdl.numeric.NumericFloat` or `com.intel.analytics.bigdl.numeric.NumericDouble`), to specify the numeric type you want.


**Scala example:**
You may check it out in the interactive Scala shell (by typing ```scala -cp bigdl_SPARKVERSION-BIGDLVERSION-SNAPSHOT-jar-with-dependencies.jar```), for instance:

```scala
 scala> import com.intel.analytics.bigdl.tensor.Tensor
 import com.intel.analytics.bigdl.tensor.Tensor
 
 scala> import com.intel.analytics.bigdl.numeric.NumericFloat
 import com.intel.analytics.bigdl.numeric.NumericFloat

 scala> val tensor = Tensor(2, 3)
 tensor: com.intel.analytics.bigdl.tensor.Tensor =
 0.0     0.0     0.0
 0.0     0.0     0.0
 [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Tensor can be created with existing data.
```scala
scala> val a = Tensor(Array(1f, 2, 3, 4, 5, 6), Array(2, 3))
a: com.intel.analytics.bigdl.tensor.Tensor[Float] =
1.0	2.0	3.0
4.0	5.0	6.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]


scala> val b = Tensor(Array(6f, 5, 4, 3, 2, 1), Array(2, 3))
b: com.intel.analytics.bigdl.tensor.Tensor[Float] =
6.0	5.0	4.0
3.0	2.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x3]
```
`+` `-` `*` `/` can be applied to tensors.
```scala
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


