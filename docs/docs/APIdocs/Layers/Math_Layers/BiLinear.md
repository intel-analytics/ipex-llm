## BiLinear

**Scala:**
```scala
val layer = BiLinear(
  inputSize1,
  inputSize2,
  outputSize,
  biasRes = true,
  wRegularizer = null,
  bRegularizer = null)
```
**Python:**
```python
layer = BiLinear(
    input_size1,
    input_size2,
    output_size,
    bias_res=True,
    wRegularizer=None,
    bRegularizer=None)
```

A bilinear transformation with sparse inputs.
The input tensor given in forward(input) is a table containing both inputs x_1 and x_2,
which are tensors of size N x inputDimension1 and N x inputDimension2, respectively.

**Parameters:**

**inputSize1**   dimension of input x_1

**inputSize2**   dimension of input x_2

**outputSize**   output dimension

**biasRes**  The layer can be trained without biases by setting bias = false. otherwise true

**wRegularizer** : instance of `Regularizer`
             (eg. L1 or L2 regularization), applied to the input weights matrices.

**bRegularizer** : instance of `Regularizer`
             applied to the bias.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Bilinear
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = Bilinear(3, 2, 3)
val input1 = Tensor(T(
  T(-1f, 2f, 3f),
  T(-2f, 3f, 4f),
  T(-3f, 4f, 5f)
))
val input2 = Tensor(T(
  T(-2f, 3f),
  T(-1f, 2f),
  T(-3f, 4f)
))
val input = T(input1, input2)

val gradOutput = Tensor(T(
  T(3f, 4f, 5f),
  T(2f, 3f, 4f),
  T(1f, 2f, 3f)
))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
-0.14168167	-8.697224	-10.097688
-0.20962894	-7.114827	-8.568602
0.16706467	-19.751905	-24.516418
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

println(grad)
 {
	2: 13.411718	-18.695072
	   14.674414	-19.503393
	   13.9599	-17.271534
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2]
	1: -5.3747015	-17.803686	-17.558662
	   -2.413877	-8.373887	-8.346823
	   -2.239298	-11.249412	-14.537216
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]
 }
```

**Python example:**
```python
layer = Bilinear(3, 2, 3)
input_1 = np.array([
  [-1.0, 2.0, 3.0],
  [-2.0, 3.0, 4.0],
  [-3.0, 4.0, 5.0]
])

input_2 = np.array([
  [-3.0, 4.0],
  [-2.0, 3.0],
  [-1.0, 2.0]
])

input = [input_1, input_2]

gradOutput = np.array([
  [3.0, 4.0, 5.0],
  [2.0, 3.0, 4.0],
  [1.0, 2.0, 5.0]
])

output = layer.forward(input)
grad = layer.backward(input, gradOutput)

print output
[[-0.5  1.5  2.5]
 [-1.5  2.5  3.5]
 [-2.5  3.5  4.5]]
[[ 3.  4.  5.]
 [ 2.  3.  4.]
 [ 1.  2.  5.]]

print grad
[array([[ 11.86168194, -14.02727222,  -6.16624403],
       [  6.72984409,  -7.96572971,  -2.89302039],
       [  5.52902842,  -5.76724434,  -1.46646953]], dtype=float32), array([[ 13.22105694,  -4.6879468 ],
       [ 14.39296341,  -6.71434498],
       [ 20.93929482, -13.02455521]], dtype=float32)]
```