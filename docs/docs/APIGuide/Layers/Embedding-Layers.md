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

