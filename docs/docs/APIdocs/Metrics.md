
---
ValidationMethod is a method to validate the model during model trainning or evaluation.
The trait can be extended by user-defined method. Now we have defined Top1Accuracy, Top5Accuracy, Loss.

---
## Loss ####

**Scala:**
```scala
val loss = new Loss(criterion)
```
**Python:**
```python
loss = Loss(cri)
```

Calculate loss of output and target with criterion. The default criterion is ClassNLLCriterion.


---
## Top1Accuracy ##

Caculate the percentage that output's max probability index equals target.

**Scala:**
```scala
val top1accuracy = new Top1Accuracy()
```
**Python:**
```python
top1accuracy = Top1Accuracy()
```

---
## Top5Accuracy ##

Caculate the percentage that target in output's top5 probability indexes.

**Scala:**
```scala
val top5accuracy = new Top5Accuracy()
```
**Python:**
```python
top5accuracy = Top5Accuracy()
```

---
## Scala Example ##

```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.models.lenet.LeNet5

val conf = Engine.createSparkConf()
val sc = new SparkContext(conf)
Engine.init
      
val data = new Array[Sample[Float]](10)
var i = 0
while (i < data.length) {
  val input = Tensor[Float](28, 28).fill(0.8f)
  val label = Tensor[Float](1).fill(1.0f)
  data(i) = Sample(input, label)
  i += 1
}
val model = LeNet5(classNum = 10)
val dataSet = sc.parallelize(data, 4)

val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](), new Loss[Float]()))
```
result is

```
result: Array[(com.intel.analytics.bigdl.optim.ValidationResult, com.intel.analytics.bigdl.optim.ValidationMethod[Float])] = Array((Accuracy(correct: 0, count: 10, accuracy: 0.0),Top1Accuracy), (Accuracy(correct: 10, count: 10, accuracy: 1.0),Top5Accuracy), ((Loss: 9.21948, count: 4, Average Loss: 2.30487),Loss))
```

## Python Example:

```
from pyspark.context import SparkContext
from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *

sc = get_spark_context(conf=create_spark_conf())
init_engine()

data_len = 10
batch_size = 8
FEATURES_DIM = 4

def gen_rand_sample():
    features = np.random.uniform(0, 1, (FEATURES_DIM))
    label = features.sum() + 0.4
    return Sample.from_ndarray(features, label)

trainingData = sc.parallelize(range(0, data_len)).map(
    lambda i: gen_rand_sample())

model = Sequential()
model.add(Linear(4, 5))
test_results = model.test(trainingData, batch_size, [Top1Accuracy(), Top5Accuracy(), Loss()])
```
result is
```
>>> print test_results[0]
Test result: 0.0, total_num: 10, method: Top1Accuracy
>>> print test_results[1]
Test result: 0.0, total_num: 10, method: Top5Accuracy
>>> print test_results[2]
Test result: 0.116546951234, total_num: 10, method: Loss
```


