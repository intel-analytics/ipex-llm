## SpatialZeroPadding ##

**Scala:**
```scala
val spatialZeroPadding = SpatialZeroPadding(padLeft, padRight, padTop, padBottom)
```
**Python:**
```python
spatialZeroPadding = SpatialZeroPadding(pad_left, pad_right, pad_top, pad_bottom)
```

Each feature map of a given input is padded with specified number of zeros.
 
If padding values are negative, then input will be cropped.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
val spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
val input = Tensor(3, 3, 3).rand()
> print(input)
(1,.,.) =
0.9494078	0.31556255	0.8432871	
0.0064580487	0.6487367	0.151881	
0.8822722	0.3634125	0.7034494	

(2,.,.) =
0.32691675	0.07487922	0.08813124	
0.4564806	0.37191486	0.05507739	
0.10097649	0.6589037	0.8721945	

(3,.,.) =
0.068939745	0.040364727	0.4893642	
0.39481318	0.17923461	0.15748173	
0.87117475	0.9933199	0.6097995

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x3x3]

>  print(spatialZeroPadding.forward(input))
(1,.,.) =
0.0	0.0064580487	0.6487367	0.151881	
0.0	0.8822722	0.3634125	0.7034494	

(2,.,.) =
0.0	0.4564806	0.37191486	0.05507739	
0.0	0.10097649	0.6589037	0.8721945	

(3,.,.) =
0.0	0.39481318	0.17923461	0.15748173	
0.0	0.87117475	0.9933199	0.6097995	

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x2x4]

```

**Python example:**
```python
from bigdl.nn.layer import *
spatialZeroPadding = SpatialZeroPadding(1, 0, -1, 0)
> spatialZeroPadding.forward(np.array([[[1, 2],[3, 4]],[[1, 2],[3, 4]]]))
[array([[[ 0.,  3.,  4.]],
       [[ 0.,  3.,  4.]]], dtype=float32)]
       
```



## Padding ##

**Scala:**
```scala
val module = Padding(dim,pad,nInputDim,value=0.0,nIndex=1)
```
**Python:**
```python
module = Padding(dim,pad,n_input_dim,value=0.0,n_index=1,bigdl_type="float")
```

This module adds pad units of padding to dimension dim of the input. If pad is negative,
padding is added to the left, otherwise, it is added to the right of the dimension.
The input to this layer is expected to be a tensor, or a batch of tensors;
when using mini-batch, a batch of sample tensors will be passed to the layer and
the user need to specify the number of dimensions of each sample tensor in the
batch using nInputDims.

 * @param dim the dimension to be applied padding operation
 * @param pad num of the pad units
 * @param nInputDim specify the number of dimensions that this module will receive
                  If it is more than the dimension of input tensors, the first dimension
                  would be considered as batch size
 * @param value padding value, default is 0

**Scala example:**
```scala
val module = Padding(1,-1,3,value=0.0,nIndex=1)
val input = Tensor(3,2,1).rand()
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.673425
0.9350421

(2,.,.) =
0.35407698
0.52607465

(3,.,.) =
0.7226349
0.70645845

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 3x2x1]

module.forward(input)
res14: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
0.0
0.0

(2,.,.) =
0.673425
0.9350421

(3,.,.) =
0.35407698
0.52607465

(4,.,.) =
0.7226349
0.70645845

[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x2x1]

```

**Python example:**
```python
module = Padding(1, -1, 3, value=0.0,n_index=1,bigdl_type="float")
input = np.random.rand(3, 2, 1)
array([[[ 0.81505274],
        [ 0.55769512]],

       [[ 0.13193961],
        [ 0.32610741]],

       [[ 0.29855582],
        [ 0.47394154]]])

module.forward(input)
array([[[ 0.        ],
        [ 0.        ]],

       [[ 0.81505275],
        [ 0.55769515]],

       [[ 0.1319396 ],
        [ 0.32610741]],

       [[ 0.29855582],
        [ 0.47394153]]], dtype=float32)
```

