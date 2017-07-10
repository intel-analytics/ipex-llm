## SpatialShareConvolution ##

**Scala:**
```scala
val layer = SpatialShareConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
```
**Python:**
```python
layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
```

 Applies a 2D convolution over an input image composed of several input planes.
 The input tensor in forward(input) is expected to be
 a 3D tensor (nInputPlane x height x width).

 This layer has been optimized to save memory. If using this layer to construct multiple convolution
 layers, please add sharing script for the fInput and fGradInput. Please refer to the ResNet example.

**Scala example:**
```scala
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = SpatialShareConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Float](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Float](Storage(biasData), 1, Array(nOutputPlane)))

    val input = Tensor[Float](Storage(inputData), 1, Array(3, 1, 3, 4))
    val output = layer.updateOutput(input)
 
    > output
res2: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0

(2,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0

(3,1,.,.) =
49.0    63.0    38.0
91.0    105.0   56.0
```

**Python example:**
```python
nInputPlane = 1
nOutputPlane = 1
kW = 2
kH = 2
dW = 1
dH = 1
padW = 0
padH = 0
layer = SpatialShareConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

input = np.array([
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1]
    ).astype("float32").reshape(3, 1, 3, 4)
layer.forward(input)

> print (output)
array([[[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]],


       [[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]],


       [[[-3.55372381, -4.0352459 , -2.65861344],
         [-4.99829054, -5.4798131 , -3.29477644]]]], dtype=float32)
```
