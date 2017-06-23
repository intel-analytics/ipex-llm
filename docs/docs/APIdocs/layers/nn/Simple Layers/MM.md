## MM ##

**Scala:**
```scala
val m = MM[T](transA,transB)
```
**Python:**
```python
m = MM(trans_a=False,trans_b=False)
```


MM is a module that performs matrix multiplication on two mini-batch inputs, producing one mini-batch.

**Scala example:**
```scala
scala> import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Tensor

scala> val input1 = Tensor[Double](3, 3).randn()
input1: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.33005356046941803     -0.07133566759754373    0.05536829452879413
1.0303293572634165      -0.15012021729192965    -0.08028050633121839
0.3629008132461407      -0.6745486660318222     -0.3050064227992411
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]

scala> val input2 = Tensor[Double](3, 3).randn()
input2: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-1.3057598149580452     0.5381646377557673      0.5735132633784201
-0.2851610351201496     -0.9109467904336456     -0.26243469354703053
0.6654744830248029      0.4493973747597665      -0.09718550515980719
[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]

scala> import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.T

scala> val input = T(1 -> input1, 2 -> input2)
input: com.intel.analytics.bigdl.utils.Table =
 {
        2: -1.3057598149580452  0.5381646377557673      0.5735132633784201
           -0.2851610351201496  -0.9109467904336456     -0.26243469354703053
           0.6654744830248029   0.4493973747597665      -0.09718550515980719
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]
        1: 0.33005356046941803  -0.07133566759754373    0.05536829452879413
           1.0303293572634165   -0.15012021729192965    -0.08028050633121839
           0.3629008132461407   -0.6745486660318222     -0.3050064227992411
           [com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x3]
 }

scala> val m1 = MM[Double]()
m1: com.intel.analytics.bigdl.nn.MM[Double] = MM()

scala> m1.forward(input)
res4: com.intel.analytics.bigdl.tensor.Tensor[Double] =
-0.3737823360541745     0.26748851845761507     0.20263005294579903
-1.3559788627784288     0.6551605066524494      0.6381064068212379
-0.48448029443942375    0.6727092413240219      0.41479560541682625
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

scala> val m2 = MM[Double](true,true)
m2: com.intel.analytics.bigdl.nn.MM[Double] = MM()

scala> m2.forward(input)
res3: com.intel.analytics.bigdl.tensor.Tensor[Double] =
0.33164457896348454     -1.1279313997494393     0.6474008319577473
-0.3745047510001215     0.33411865538700675     -0.049379345201886776
-0.2904270525180214     0.13738665649333798     0.030410541664609925
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x3]

```

**Python example:**
```python
from bigdl.nn.layer import *
import numpy as np

input1=np.random.rand(3,3)
input2=np.random.rand(3,3)
input = [input1,input2]
print "input is :",input
out = MM().forward(input)
print "output is :",out
```
produces output:
```python
input is : [array([[ 0.13696046,  0.92653165,  0.73585328],
       [ 0.28167852,  0.06431783,  0.15710073],
       [ 0.21896166,  0.00780161,  0.25780671]]), array([[ 0.11232797,  0.17023931,  0.92430042],
       [ 0.86629537,  0.07630215,  0.08584417],
       [ 0.47087278,  0.22992833,  0.59257503]])]
creating: createMM
output is : [array([[ 1.16452789,  0.26320592,  0.64217824],
       [ 0.16133308,  0.08898225,  0.35897085],
       [ 0.15274818,  0.09714822,  0.3558259 ]], dtype=float32)]
```