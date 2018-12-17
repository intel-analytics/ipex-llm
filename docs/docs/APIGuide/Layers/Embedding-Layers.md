## LookupTable ##

**Scala:**
```scala
val layer = LookupTable(nIndex: Int, nOutput: Int, paddingValue: Double = 0,
                                 maxNorm: Double = Double.MaxValue,
                                 normType: Double = 2.0,
                                 shouldScaleGradByFreq: Boolean = false,
                                 wRegularizer: Regularizer[T] = null)
```

**Python:**
```python
layer = LookupTable(nIndex, nOutput, paddingValue, maxNorm, normType, shouldScaleGradByFreq)
```

This layer is a particular case of a convolution, where the width of the convolution would be 1.
Input should be a 1D or 2D tensor filled with indices. Indices are corresponding to the position
in weight. For each index element of input, it outputs the selected index part of weight.
This layer is often used in word embedding. In collaborative filtering, it can be used together with Select to create embeddings for users or items. 

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._

val layer = LookupTable(9, 4, 2, 0.1, 2.0, true)
val input = Tensor(Storage(Array(5.0f, 2.0f, 6.0f, 9.0f, 4.0f)), 1, Array(5))

val output = layer.forward(input)
val gradInput = layer.backward(input, output)

> println(layer.weight)
res6: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.2949163      -0.8240777      -0.9440595      -0.8326071
-0.025108865    -0.025346711    0.09046136      -0.023320194
-1.7525806      0.7305201       0.3349018       0.03952092
-0.0048129847   0.023922665     0.005595926     -0.09681542
-0.01619357     -0.030372608    0.07217587      -0.060049288
0.014426847     -0.09052222     0.019132217     -0.035093457
-0.7002858      1.1149521       0.9869375       1.2580993
0.36649692      -0.6583153      0.90005803      0.12671651
0.048913725     0.033388995     -0.07938445     0.01381052
[com.intel.analytics.bigdl.tensor.DenseTensor of size 9x4]

> println(input)
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
5.0
2.0
6.0
9.0
4.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]

> println(output)
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.01619357     -0.030372608    0.07217587      -0.060049288
-0.025108865    -0.025346711    0.09046136      -0.023320194
0.014426847     -0.09052222     0.019132217     -0.035093457
0.048913725     0.033388995     -0.07938445     0.01381052
-0.0048129847   0.023922665     0.005595926     -0.09681542
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5x4]

> println(gradInput)
gradInput: com.intel.analytics.bigdl.tensor.Tensor[Float] =
0.0
0.0
0.0
0.0
0.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 5]

```

**Python example:**
```python
layer = LookupTable(9, 4, 2.0, 0.1, 2.0, True)
input = np.array([5.0, 2.0, 6.0, 9.0, 4.0]).astype("float32")

output = layer.forward(input)
gradInput = layer.backward(input, output)

> output
[array([[-0.00704637,  0.07495038,  0.06465427,  0.01235369],
        [ 0.00350313,  0.02751033, -0.02163727,  0.0936095 ],
        [ 0.02330465, -0.05696457,  0.0081728 ,  0.07839092],
        [ 0.06580321, -0.0743262 , -0.00414508, -0.01133001],
        [-0.00382435, -0.04677011,  0.02839171, -0.08361723]], dtype=float32)]

> gradInput
[array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)]

```

---
## LookupTableSparse ##

**Scala:**
```scala
val layer = LookupTableSparse(nIndex: Int, nOutput: Int,
    combiner: String = "sum",
    maxNorm: Double = -1,
    wRegularizer: Regularizer[T] = null)
```

**Python:**
```python
layer = LookupTableSparse(nIndex, nOutput,
    combiner,
    maxNorm,
    wRegularizer)
```

LookupTable for multi-values. 
Also called embedding_lookup_sparse in TensorFlow. 

The input of LookupTableSparse should be a 2D SparseTensor or two 2D SparseTensors. If the input is a SparseTensor, the values are positive integer ids, values in each row of this SparseTensor will be turned into a dense vector. If the input is two SparseTensor, the first tensor should be the integer ids, just like the SparseTensor input. And the second tensor is the corresponding weights of the integer ids.

@param nIndex Indices of input row
@param nOutput the last dimension size of output
@param combiner A string specifying the reduce type. Currently "mean", "sum", "sqrtn" is supported.
@param maxNorm If provided, each embedding is normalized to have l2 norm equal to maxNorm before combining.
@param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the input weights matrices.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T

val indices1 = Array(0, 0, 1, 2)
val indices2 = Array(0, 1, 0, 3)
val values = Array(2f, 4, 1, 2)
val weightValues = Array(2f, 0.5f, 1, 3)
val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

val layer1 = LookupTableSparse(10, 4, "mean")
layer1.weight.range(1, 40, 1) // set weight to 1 to 40
val output = layer1.forward(T(input, weight))
```
The output is
```scala
output: com.intel.analytics.bigdl.tensor.Tensor[Float] =
6.6	7.6000004	8.6	9.6
1.0	2.0	3.0	4.0
5.0	6.0	7.0	8.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4]
```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

indices = np.array([[0, 0, 1, 2], [0, 1, 0, 3]])
values = np.array([2, 4, 1, 2])
weightValues = np.array([2, 0.5, 1, 3])
input = JTensor.sparse(values, indices, np.array([3, 4]))
weight = JTensor.sparse(weightValues, indices, np.array([3, 4]))

layer1 = LookupTableSparse(10, 4, "mean")
layer1.set_weights(np.arange(1, 41, 1).reshape(10, 4)) # set weight to 1 to 40
output = layer1.forward([input, weight])
print(output)
```
The output is
```python
array([[ 6.5999999 ,  7.60000038,  8.60000038,  9.60000038],
       [ 1.        ,  2.        ,  3.        ,  4.        ],
       [ 5.        ,  6.        ,  7.        ,  8.        ]], dtype=float32)
```

