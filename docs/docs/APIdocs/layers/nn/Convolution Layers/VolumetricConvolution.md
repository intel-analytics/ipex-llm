## VolumetricConvolution ##

**Scala:**
```scala
val module = VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH)
```
**Python:**
```python
module = VolumetricConvolution(n_input_plane, n_output_plane, k_t, k_w, k_h)
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
 * @param withBias whether with bias
 * @param initMethod Init method, Default, Xavier, Bilinear.
 
**Scala example:**
```scala
val layer = VolumetricConvolution[Double](3, 4, 4, 3, 2, 3, 2, 2, 1, 1, 1)
val input = Tensor[Double](3, 5, 6, 6).rand()

layer.forward(input)
(1,1,.,.) =
-0.1914729317950074     -0.08113735619836425    0.10788965883728818
0.10395020474256289     0.13462794360815336     0.01795723291533359
0.12943407582902128     -0.1690309245070469     0.005821897452482808
-0.011252358426331377   -0.11427828750672125    -0.08924567153406285

(1,2,.,.) =
0.07354218378093497     -0.076352514893191      -0.16656841203957992
0.1474835835920596      0.2558052730720035      -0.08918594404456118
0.10126311292469131     0.1836231848375178      -0.07782036577883622
0.20178209579248316     -0.15767076220928622    -0.04782003126825497

(2,1,.,.) =
-0.03242712780702128    -0.3649814573965776     -0.041354367554219396
-0.093357235374353      -0.48948518373955563    -0.4277957331255818
-0.13937510329639902    -0.3578734836823814     -0.449763889250075
-0.22252364458108947    ...
```

**Python example:**
```python
layer = VolumetricConvolution(3, 4, 4, 3, 2, 3, 2, 2, 1, 1, 1)
input = np.random.rand(3,5,6,6)
[array([[
[[ 0.01081296,  0.26384461,  0.1281845 ],
 [ 0.12007203, -0.08885437,  0.19182643],
 [-0.0004613 ,  0.1039933 ,  0.19885567],
 [-0.12519719, -0.1839595 , -0.25746709]],

[[ 0.11065411,  0.32261357,  0.24915619],
 [ 0.00798186,  0.13440473,  0.38609582],
 [ 0.25891161,  0.07706913,  0.44633675],
 [ 0.10668587, -0.17029612, -0.05370832]]],


[[[-0.2126165 , -0.01200591, -0.01666662],
 [-0.14056835,  0.12342533,  0.06915439],
 [ 0.00849888,  0.05986062, -0.03426152],
 [-0.09622344,  0.04998226, -0.01080746]],

[[-0.18693367, -0.2059519 ,  0.05560745],
 [-0.08661881, -0.33304274,  0.03641717],
 [-0.04102507,  0.3228099 ,  0.3916195 ],
 [-0.00589391,  0.02847509,  0.04867363]]],


[[[-0.17219372, -0.25455672, -0.06510431],
 [-0.18989559,  0.02380426,  0.23785897],
 [-0.18200465,  0.19869429,  0.37195116],
 [ 0.1671378 ,  0.24541213,  0.26872626]],

[[-0.03391223, -0.05975001, -0.05230101],
 [-0.03596913,  0.04396246, -0.09864822],
 [ 0.0556287 ,  0.10142146, -0.20651391],
 [ 0.18441495,  0.2497469 ,  0.07067506]]],


[[[-0.2139709 , -0.19772246, -0.01539564],
 [-0.08242216,  0.01481231, -0.00846959],
 [-0.01496931, -0.17518835,  0.05056693],
 [ 0.2369837 ,  0.22532576,  0.06207009]],

[[-0.25084218, -0.17404027,  0.10309691],
 [-0.14596771, -0.35834655, -0.18897848],
 [-0.29150304, -0.2739552 , -0.30588183],
 [ 0.05461492,  0.00377342,  0.01728923]]]], dtype=float32)]
```
