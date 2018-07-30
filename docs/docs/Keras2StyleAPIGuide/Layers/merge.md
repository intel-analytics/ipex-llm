## **Maximum**
Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

**Scala:**
```scala
Maximum()
```
**Python:**
```python
Maximum()
```

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Maximum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = Maximum[Float]().inputs(Array(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.0085953	1.1095089	1.7487661	1.576811	1.3192933	1.173145	1.7567515	1.750411	
	   1.0303572	1.0285444	1.4724362	1.0070276	1.6837391	1.2812499	1.7207997	1.9301186	
	   1.6642286	1.300531	1.2989123	1.0117699	1.5870146	1.2845709	1.9443712	1.1186409	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.6898564	0.98180896	0.22475463	0.44690642	0.22631128	0.8658154	0.96297216	0.038640756	
	   0.33791444	0.35920507	0.2056811	0.97009206	0.891668	0.73843783	0.49456882	0.92106706	
	   0.54771185	0.52310455	0.49114317	0.93534994	0.82244986	0.080847055	0.56450963	0.73846775	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
1.0085953	1.1095089	1.7487661	1.576811	1.3192933	1.173145	1.7567515	1.750411	
1.0303572	1.0285444	1.4724362	1.0070276	1.6837391	1.2812499	1.7207997	1.9301186	
1.6642286	1.300531	1.2989123	1.0117699	1.5870146	1.2845709	1.9443712	1.1186409	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import Maximum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = Maximum()([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.85046637 0.76454759 0.92092265 0.18948392 0.96141892 0.75558563
  0.16956892 0.49839472]
 [0.36737777 0.25567011 0.36751645 0.49982497 0.62344662 0.10207675
  0.14432582 0.09316922]
 [0.34775348 0.56521665 0.01922694 0.97405856 0.96318355 0.48008106
  0.09525403 0.64539933]]

input2:
[[0.23219699 4.58298671 4.08334902 3.35729794 3.28995515 3.88572392
  0.13552906 2.20767025]
 [4.41043478 0.74315223 1.57928439 4.06317265 4.35646267 4.43969778
  0.64163024 0.14681471]
 [1.60829488 3.75488617 4.69265858 1.38504037 3.2210222  3.4321568
  4.00735856 2.6106414 ]]
```
Output is
```python
[[0.8504664  4.582987   4.083349   3.357298   3.2899551  3.8857238
  0.16956893 2.2076702 ]
 [4.4104347  0.74315226 1.5792844  4.063173   4.3564625  4.4396977
  0.64163023 0.1468147 ]
 [1.6082948  3.7548862  4.6926584  1.3850404  3.2210221  3.4321568
  4.0073586  2.6106415 ]]
```

---
## **maximum**
Functional interface to the `Maximum` layer.

**Scala:**
```scala
maximum(inputs)
```
**Python:**
```python
maximum(inputs)
```

**Parameters:**

* `inputs`: A list of input tensors (at least 2).
* `**kwargs`: Standard layer keyword arguments.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Maximum.maximum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = maximum(inputs = List(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.5386189	1.67534	1.3651735	1.0366004	1.2869223	1.6384993	1.5557045	1.5723307	
	   1.2382979	1.0155076	1.1055984	1.1010389	1.6874355	1.3107576	1.2041453	1.9931196	
	   1.4011493	1.0774659	1.3888124	1.7762307	1.8265619	1.7934192	1.7732148	1.2978737	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.14446749	0.7428541	0.9886685	0.5107685	0.85201174	0.40988243	0.12447342	0.8556565	
	   0.91737056	0.35073906	0.07863916	0.89909834	0.8177192	0.09691833	0.1997524	0.4406145	
	   0.4190805	0.6956053	0.9765333	0.6748145	0.87814146	0.5421859	0.31012502	0.25200275	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
1.5386189	1.67534	1.3651735	1.0366004	1.2869223	1.6384993	1.5557045	1.5723307	
1.2382979	1.0155076	1.1055984	1.1010389	1.6874355	1.3107576	1.2041453	1.9931196	
1.4011493	1.0774659	1.3888124	1.7762307	1.8265619	1.7934192	1.7732148	1.2978737	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import maximum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = maximum([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.32837152 0.66842081 0.5893283  0.71063029 0.53254716 0.98882168
  0.53400631 0.93659819]
 [0.6198554  0.51117444 0.74729989 0.65475831 0.70510429 0.87443468
  0.5629698  0.285089  ]
 [0.43159809 0.84360242 0.8493521  0.78723246 0.35496674 0.00144353
  0.07231955 0.76153367]]
  
 input2: 
[[4.00763759 0.37730923 3.88563172 2.22099527 3.38980926 2.84321074
  0.29846632 4.07808143]
 [0.36804983 2.34995472 2.24190514 1.63816757 2.22642342 1.45099988
  0.55931613 0.42101343]
 [0.30218586 2.75409562 0.24024987 3.89805855 4.57479762 2.6592906
  2.38562566 1.46560388]]
```
Output is
```python
[[4.0076375  0.6684208  3.8856318  2.2209952  3.3898094  2.8432107
  0.5340063  4.0780816 ]
 [0.6198554  2.3499546  2.2419052  1.6381676  2.2264235  1.4509999
  0.5629698  0.42101344]
 [0.4315981  2.7540956  0.8493521  3.8980587  4.5747976  2.6592906
  2.3856256  1.4656038 ]]
```

---
## **Minimum**
Layer that computes the minimum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

**Scala:**
```scala
Minimum()
```
**Python:**
```python
Minimum()
```

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = Minimum[Float]().inputs(Array(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.9953886	1.0161483	1.844671	1.1757553	1.7548938	1.4735664	1.981268	1.354598	
	   1.786057	1.4920603	1.538079	1.6601591	1.5213481	1.9032607	1.5938802	1.9769413	
	   1.428338	1.5083437	1.1141979	1.4320385	1.9785057	1.845624	1.0637122	1.8684102	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.6624951	0.71156764	0.004659928	0.8797748	0.8676378	0.5605965	0.03135305	0.3550916	
	   0.86810714	0.26216865	0.8639284	0.3357767	0.22505952	0.8216017	0.74407136	0.73391193	
	   0.74810994	0.11495259	0.89162785	0.93693215	0.5673804	0.20798753	0.022446347	0.36790285	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
0.6624951	0.71156764	0.004659928	0.8797748	0.8676378	0.5605965	0.03135305	0.3550916	
0.86810714	0.26216865	0.8639284	0.3357767	0.22505952	0.8216017	0.74407136	0.73391193	
0.74810994	0.11495259	0.89162785	0.93693215	0.5673804	0.20798753	0.022446347	0.36790285	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import Minimum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = Minimum()([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.15979525 0.48601263 0.10587506 0.61769843 0.26736246 0.64769634
  0.01616307 0.93659085]
 [0.3412241  0.02449786 0.64638927 0.32875475 0.77737532 0.94151168
  0.95571165 0.1285685 ]
 [0.9758039  0.89746475 0.84606271 0.87471803 0.80568297 0.85872464
  0.77484317 0.73048055]]

input2:
[[0.99780609 1.48670819 0.08911578 2.68460415 1.21065202 1.82819649
  2.91991375 1.07241835]
 [3.18491884 3.72856744 3.82128444 1.53010301 1.20795887 3.20653343
  3.07794378 1.59084261]
 [4.39776482 3.37465746 0.23752302 3.47325532 2.38110537 4.64806043
  3.99013359 0.56055062]]
```
Output is
```python
[[0.15979525 0.48601264 0.08911578 0.61769843 0.26736248 0.6476963
  0.01616307 0.93659085]
 [0.3412241  0.02449786 0.64638925 0.32875475 0.77737534 0.9415117
  0.95571166 0.1285685 ]
 [0.9758039  0.89746475 0.23752302 0.874718   0.80568296 0.85872465
  0.77484316 0.56055063]]
```

---
## **minimum**
Functional interface to the `Minimum` layer.

**Scala:**
```scala
minimum(inputs)
```
**Python:**
```python
minimum(inputs)
```

**Parameters:**

* `inputs`: A list of input tensors (at least 2).
* `**kwargs`: Standard layer keyword arguments.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum.minimum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = minimum(inputs = List(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.0131017	1.7637167	1.3681185	1.6208028	1.2059574	1.967363	1.5065156	1.5110291	
	   1.1055611	1.4148856	1.5531528	1.3481603	1.3744175	1.5192658	1.7290237	1.629003	
	   1.5601189	1.4540797	1.0981613	1.2463317	1.9510872	1.0527081	1.0487831	1.4148198	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.96061057	0.41263154	0.5029265	0.28855637	0.8030459	0.5923882	0.93190056	0.15111573	
	   0.54223496	0.37586558	0.63049513	0.32910138	0.029513072	0.017590795	0.1943584	0.77225924	
	   0.21727595	0.6552713	0.899118	0.07937545	0.016797619	0.5491529	0.7383374	0.8877089	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
0.96061057	0.41263154	0.5029265	0.28855637	0.8030459	0.5923882	0.93190056	0.15111573	
0.54223496	0.37586558	0.63049513	0.32910138	0.029513072	0.017590795	0.1943584	0.77225924	
0.21727595	0.6552713	0.899118	0.07937545	0.016797619	0.5491529	0.7383374	0.8877089	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import minimum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = minimum([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.82114595 0.19940446 0.57764964 0.49092517 0.67671559 0.72676631
  0.85955214 0.80265871]
 [0.05640074 0.17010821 0.4896911  0.14905843 0.33233282 0.82684842
  0.58635163 0.10010479]
 [0.83053659 0.83788089 0.6177536  0.71670009 0.54454425 0.19431431
  0.49180683 0.25640596]]

input2:
[[0.47446558 3.8752243  4.9299194  3.35971335 0.85980843 2.37388383
  4.38802943 4.3253041 ]
 [2.65459389 2.93173369 3.6176582  0.75475853 0.62484204 4.16820336
  3.24864692 1.42238813]
 [0.439386   2.43623362 0.20248675 1.60213208 1.08081789 0.59718494
  0.29896311 0.73010527]]
```
Output is
```python
[[0.47446558 0.19940446 0.57764965 0.49092516 0.6767156  0.7267663
  0.85955215 0.80265874]
 [0.05640074 0.17010821 0.4896911  0.14905843 0.33233282 0.82684845
  0.58635163 0.10010479]
 [0.439386   0.8378809  0.20248675 0.7167001  0.5445443  0.1943143
  0.2989631  0.25640595]]
```
