## Linear ##

**Scala:**
```scala
val module = Linear(
  inputSize: Int,
  outputSize: Int,
  initMethod: InitializationMethod = Default,
  withBias: Boolean = true)
```
**Python:**
```python
module = Linear(
  input_size,
  output_size,
  init_method="default",
  with_bias=True)
```

The `Linear` module applies a linear transformation to the input data,
i.e. `y = Wx + b`. The `input` given in `forward(input)` must be either
a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
have the size of `inputSize`. If it is a matrix, then each row is assumed to be
an input sample of given batch (the number of rows means the batch size and
the number of columns should be equal to the `inputSize`).

**Scala example:**
```scala
val module = Linear(3, 5)

println(module.forward(Tensor.range(1, 3, 1)))
```
Output is
```com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.79338956
-2.3417668
-2.7557678
-0.07507719
-1.009765
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]
```

**Python example:**
```python
module = Linear(3, 5)

print(module.forward(np.arange(1, 4, 1)))
```
Output is
```
[array([ 0.31657887, -1.11062765, -1.16235781, -0.67723978,  0.74650359], dtype=float32)]
```
