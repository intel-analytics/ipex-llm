## VolumetricConvolution ##

**Scala:**
```scala
val module = VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH,
  dT=1, dW=1, dH=1, padT=0, padW=0, padH=0, withBias=true)
```
**Python:**
```python
module = VolumetricConvolution(n_input_plane, n_output_plane, k_t, k_w, k_h,
  d_t=1, d_w=1, d_h=1, pad_t=0, pad_w=0, pad_h=0, with_bias=true)
```

Applies a 3D convolution over an input image composed of several input planes. The input tensor
in forward(input) is expected to be a 4D tensor (nInputPlane x time x height x width).
 * @param nInputPlane The number of expected input planes in the image given into forward()
 * @param nOutputPlane The number of output planes the convolution layer will produce.
 * @param kT The kernel size of the convolution in time
 * @param kW The kernel width of the convolution
 * @param kH The kernel height of the convolution
 * @param dT The step of the convolution in the time dimension. Default is 1
 * @param dW The step of the convolution in the width dimension. Default is 1
 * @param dH The step of the convolution in the height dimension. Default is 1
 * @param padT Additional zeros added to the input plane data on both sides of time axis.
 * Default is 0. (kT-1)/2 is often used here.
 * @param padW The additional zeros added per width to the input planes.
 * @param padH The additional zeros added per height to the input planes.
 * @param withBias whether with bias.
 
**Scala example:**
```scala
val layer = VolumetricConvolution(2, 3, 2, 2, 2, dT=1, dW=1, dH=1,
  padT=0, padW=0, padH=0, withBias=true)
val input = Tensor(2, 2, 2, 2).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.54846555      0.5549177
0.43748873      0.6596535

(1,2,.,.) =
0.87915933      0.5955469
0.67464 0.40921077

(2,1,.,.) =
0.24127467      0.49356017
0.6707502       0.5421975

(2,2,.,.) =
0.007834963     0.08188637
0.51387626      0.7376101

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x2x2x2]

layer.forward(input)
res16: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.6680023

(2,1,.,.) =
0.41926455

(3,1,.,.) =
-0.029196609

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x1x1]
```

**Python example:**
```python
layer = VolumetricConvolution(2, 3, 2, 2, 2, d_t=1, d_w=1, d_h=1,
          pad_t=0, pad_w=0, pad_h=0, with_bias=True, init_method="default",
          bigdl_type="float")
input = np.random.rand(2,2,2,2)
 array([[[[ 0.47639062,  0.76800312],
         [ 0.28834351,  0.21883535]],

        [[ 0.86097919,  0.89812597],
         [ 0.43632181,  0.58004824]]],


       [[[ 0.65784027,  0.34700039],
         [ 0.64511955,  0.1660241 ]],

        [[ 0.36060054,  0.71265665],
         [ 0.51755249,  0.6508298 ]]]])
 
layer.forward(input)
array([[[[ 0.54268712]]],


       [[[ 0.17670505]]],


       [[[ 0.40953237]]]], dtype=float32)

```
