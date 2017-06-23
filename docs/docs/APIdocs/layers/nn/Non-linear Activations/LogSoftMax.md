## LogSoftMax ##

**Scala:**
```scala
val model = LogSoftMax()
```
**Python:**
```python
model = LogSoftMax()
```

The LogSoftMax module applies a LogSoftMax transformation to the input data
which is defined as:
```
f_i(x) = log(1 / a exp(x_i))
where a = sum_j[exp(x_j)]
```
The input given in `forward(input)` must be either
a vector (1D tensor) or matrix (2D tensor).

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

val model = LogSoftMax()
val input = Tensor(2, 5).rand()
val output = model.forward(input)

scala> print(input)
0.4434036	0.64535594	0.7516194	0.11752353	0.5216674	
0.57294756	0.744955	0.62644184	0.0052207764	0.900162	
[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 2x5]

scala> print(output)
-1.6841899	-1.4822376	-1.3759742	-2.01007	-1.605926	
-1.6479948	-1.4759872	-1.5945004	-2.2157214	-1.3207803	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x5]
```

**Python example:**
```python
model = LogSoftMax()
input = np.random.randn(4, 10)
output = model.forward(input)

>>> print(input)
[[ 0.10805365  0.11392282  1.31891713 -0.62910637 -0.80532589  0.57976863
  -0.44454368  0.26292944  0.8338328   0.32305099]
 [-0.16443839  0.12010763  0.62978233 -1.57224143 -2.16133614 -0.60932395
  -0.22722708  0.23268273  0.00313597  0.34585582]
 [ 0.55913444 -0.7560615   0.12170887  1.40628806  0.97614582  1.20417145
  -1.60619173 -0.54483025  1.12227399 -0.79976189]
 [-0.05540945  0.86954458  0.34586427  2.52004267  0.6998163  -1.61315173
  -0.76276874  0.38332142  0.66351792 -0.30111399]]

>>> print(output)
[[-2.55674744 -2.55087829 -1.34588397 -3.2939074  -3.47012711 -2.08503246
  -3.10934472 -2.40187168 -1.83096838 -2.34175014]
 [-2.38306785 -2.09852171 -1.58884704 -3.79087067 -4.37996578 -2.82795334
  -2.44585633 -1.98594666 -2.21549344 -1.87277353]
 [-2.31549931 -3.63069534 -2.75292492 -1.46834576 -1.89848804 -1.67046237
  -4.48082542 -3.41946411 -1.75235975 -3.67439556]
 [-3.23354769 -2.30859375 -2.83227396 -0.6580956  -2.47832203 -4.79128981
  -3.940907   -2.79481697 -2.5146203  -3.47925234]]
```
