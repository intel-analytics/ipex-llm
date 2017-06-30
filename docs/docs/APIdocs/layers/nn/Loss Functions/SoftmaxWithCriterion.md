## SoftmaxWithCriterion ##

**Scala:**
```scala
val model = SoftmaxWithCriterion(ignoreLabel, normalizeMode)
```
**Python:**
```python
model = SoftmaxWithCriterion(ignoreLabel, normalizeMode)
```

Computes the multinomial logistic loss for a one-of-many classification task, passing real-valued predictions through a softmax to
get a probability distribution over classes. It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer as 
its gradient computation is more numerically stable.

- param ignoreLabel   (optional) Specify a label value that should be ignored when computing the loss.
- param normalizeMode How to normalize the output loss.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

val input = Tensor(1, 5, 2, 3).rand()
val target = Tensor(Storage(Array(2.0f, 4.0f, 2.0f, 4.0f, 1.0f, 2.0f))).resize(1, 1, 2, 3)

val model = SoftmaxWithCriterion[Float]()
val output = model.forward(input, target)

scala> print(input)
(1,1,.,.) =
0.65131104	0.9332143	0.5618989	
0.9965054	0.9370902	0.108070895	

(1,2,.,.) =
0.46066576	0.9636703	0.8123812	
0.31076035	0.16386998	0.37894428	

(1,3,.,.) =
0.49111295	0.3704862	0.9938375	
0.87996656	0.8695406	0.53354675	

(1,4,.,.) =
0.8502225	0.9033509	0.8518651	
0.0692618	0.10121379	0.970959	

(1,5,.,.) =
0.9397213	0.49688303	0.75739735	
0.25074655	0.11416598	0.6594504	

[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x5x2x3]

scala> print(output)
1.6689054
```
**Python example:**
```python
input = np.random.randn(1, 5, 2, 3)
target = np.array([[[[2.0, 4.0, 2.0], [4.0, 1.0, 2.0]]]])

model = SoftmaxWithCriterion()
output = model.forward(input, target)

>>> print input
[[[[ 0.78455689  0.01402084  0.82539628]
   [-1.06448238  2.58168413  0.60053703]]

  [[-0.48617618  0.44538094  0.46611658]
   [-1.41509329  0.40038991 -0.63505732]]

  [[ 0.91266769  1.68667933  0.92423611]
   [ 0.1465411   0.84637557  0.14917515]]

  [[-0.7060493  -2.02544114  0.89070726]
   [ 0.14535539  0.73980064 -0.33130613]]

  [[ 0.64538791 -0.44384233 -0.40112523]
   [ 0.44346658 -2.22303621  0.35715986]]]]

>>> print output
2.1002123

```
