# Weight Initialization

## Usage

After creating a new layer, you can use the `setInitMethod` method to
specify the weight and bias initialization method. The initialization
methods will be used to initialize weight and bias respectively and they
will be called in `model.reset()`. `model.reset()` will be called inside
`setInitMethod`, so you do not have to call it again after setting the method.

* scala code
```scala
val weightInitMethod = Xavier
val biasInitMethod = Zeros
val model = Linear(10, 20)
model.setInitMethod(weightInitMethod, biasInitMethod)
```

* python code
```python
weight_init_method = Xavier()
bias_init_method = Zeros()
model = Linear(10, 20)
model.set_init_method(weight_init_method, bias_init_method)
```

## Pre-defined Weight Initializers
### Uniform
This initialization method draws samples from a uniform distribution.
If the lower bound and upper bound of this uniform distribution is not
specified, it will be set to [-limit, limit) where limit = 1/sqrt(fanIn).

* scala code
```scala
val uniformInitMethod1 = RandomUniform // U(-limit, limit) where limit = 1/sqrt(fanIn)
val (lower, upper) = (0.0, 1.0)
val uniformInitMethod2 = RandomUniform(lower, upper) // U(lower, upper)
```

* python code
```python
uniformInitMethod1 = RandomUniform() # U(-limit, limit) where limit = 1/sqrt(fanIn)
(lower, upper) = (0.0, 1.0)
uniformInitMethod2 = RandomUniform(lower, upper) # U(lower, upper)
```

### Xavier
The Xavier initialization method draws samples from a uniform distribution
bounded by [-limit, limit) where limit = sqrt(6.0/(fanIn+fanOut)). The rationale
behind this formula can be found in the paper
[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).

* scala code
```scala
val initMethod = Xavier
```

* python code
```python
init_method = Xavier()
```

### Normal
This initialization method draws samples from a normal distribution.

* scala code
```scala
val initMethod = RandomNormal(mean=0.0, stdv=1.0)
```

* python code
```python
init_method = RandomNormal(mean=0.0, stdv=1.0)
```

### Ones
Initialization method that set tensor to ones.

* scala code
```scala
val initMethod = Ones
```

* python code
```python
init_method = Ones()
```

### Zeros
Initialization method that set tensor to zeros.

* scala code
```scala
val initMethod = Zeros
```

* python code
```python
init_method = Zeros()
```

### ConstInitMethod
Initialization method that set tensor to the specified constant value.

* scala code
```scala
val initMethod = ConstInitMethod(0.1)
```

* python code
```python
init_method = ConstInitMethod(0.1)
```

### BilinearFiller
Initialize the weight with coefficients for bilinear interpolation.
A common use case is with the DeconvolutionLayer acting as upsampling.
This initialization method can only be used in the weight initialization
of SpatialFullConvolution.

* scala code
```scala
val initMethod = BilinearFiller
```

* python code
```python
init_method = BilinearFiller()
```


## Define your own Weight Initializer
### Scala
All initialization methods should implement the `InitializationMethod` trait

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