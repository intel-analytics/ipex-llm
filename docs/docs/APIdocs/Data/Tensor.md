## **Tensor**

Modeled after the [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md) class in [Torch](http://torch.ch/  ), the ```Tensor``` [package](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/    analytics/bigdl/tensor) (written in Scala and leveraging [Intel MKL](https://software.intel.com/en-us/intel-mkl)) in     BigDL provides numeric computing support for the deep learning applications (e.g., the input, output, weight, bias and   gradient of the neural networks).

A ```Tensor``` is essentially a multi-dimensional array of numeric types (e.g., ```Int```, ```Float```, ```Double```,    etc.); you may check it out in the interactive Scala shell (by typing ```scala -cp bigdl_0.1-0.1.0-SNAPSHOT-jar-with-    dependencies.jar```), for instance:

```scala
 scala> import com.intel.analytics.bigdl.tensor.Tensor
 import com.intel.analytics.bigdl.tensor.Tensor

 scala> val tensor = Tensor[Float](2, 3)
 tensor: com.intel.analytics.bigdl.tensor.Tensor[Float] =
 0.0     0.0     0.0
 0.0     0.0     0.0
 [com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
