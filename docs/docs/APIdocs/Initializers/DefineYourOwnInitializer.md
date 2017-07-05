## CustomizedInitializer ##

All customizedInitializer should implement the `InitializationMethod` trait

```scala
/**
 * Initialization method to initialize bias and weight.
 * The init method will be called in Module.reset()
 */

trait InitializationMethod {

  type Shape = Array[Int]

  /**
   * Initialize the given variable
   *
   * @param variable    the variable to initialize
   * @param dataFormat  describe the meaning of each dimension of the variable
   */
  def init[T](variable: Tensor[T], dataFormat: VariableFormat)
             (implicit ev: TensorNumeric[T]): Unit
}
```
The [RandomUniform](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn/InitializationMethod.scala#L163)
code should give you a good sense of how to implement this trait.


### Python
Custom initialization method in python is not supported right now.

