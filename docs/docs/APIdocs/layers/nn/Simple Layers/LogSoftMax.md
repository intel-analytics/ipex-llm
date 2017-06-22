## LogSoftMax ##

**Scala:**
```scala
val model = LogSoftMax[T]()
```
**Python:**
```python
model = LogSoftMax()
```

The [[LogSoftMax]] module applies a LogSoftMax transformation to the input data
which is defined as:
```
f_i(x) = log(1 / a exp(x_i))
where a = sum_j[exp(x_j)]
```
The input given in `forward(input)` must be either
a vector (1D tensor) or matrix (2D tensor).

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = LogSoftMax[Float]()
val input = Tensor[Float](4, 10).rand()
val output = model.forward(input)
```
output is
```
output: com.intel.analytics.bigdl.tensor.Tensor[Float] = 
-2.5519505	-2.842981	-2.7291634	-2.5205712	-2.21346	-2.0949225	-2.0828898	-2.0750494	-1.98947	-2.3142338	
-2.4164822	-2.2851913	-2.344031	-1.9481893	-2.3196316	-1.9729276	-2.7517028	-2.2612143	-2.7797568	-2.2722216	
-2.5480394	-2.5438929	-2.1472383	-1.8264041	-2.4599571	-2.3786807	-2.347884	-1.8615696	-2.6476033	-2.726078	
-2.2741008	-2.0731382	-2.500853	-2.3554156	-2.2530231	-2.014162	-2.5651312	-2.1602802	-2.4301133	-2.5808542	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 4x10]
```

**Python example:**
```python
model = LogSoftMax()
input = np.random.randn(4, 10)
output = model.forward(input)
```
output is
```
array([[-2.71494865, -1.94257474, -3.11852407, -1.23062432, -2.46524668,
        -3.04385233, -2.50393724, -2.94146872, -3.42119932, -1.86910367],
       [-3.47281766, -1.87250924, -1.78490853, -4.42017174, -3.09390163,
        -1.90380895, -3.61078787, -1.01130319, -3.57430983, -3.80523968],
       [-1.20955157, -2.81982231, -1.93510222, -2.77889538, -3.94818759,
        -2.5525887 , -1.46124649, -4.911623  , -2.73293018, -3.38049865],
       [-2.79008269, -2.73434305, -3.92586136, -3.16623569, -2.66140938,
        -4.14181805, -1.25464666, -1.69758749, -3.08269215, -1.55173862]], dtype=float32)
```
