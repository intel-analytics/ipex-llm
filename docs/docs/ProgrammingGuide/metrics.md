ValidationMethod is a method to validate the model during model training or evaluation.
The trait can be extended by user-defined method.

In addition of the ValidationMethods provided in BigDL, Zoo provides several extra metrics for practical industry applications.

---
## AUC ####
Area under the ROC curve. More information about ROC can be found https://en.wikipedia.org/wiki/Receiver_operating_characteristic
It's used to evaluate a binary(0/1 only) classification mode. It supports single label and multiple labels.

**Scala:**
```scala
val validation = new AUC(20)
```
example
```scala
val conf = Engine.createSparkConf()
  .setAppName("AUC test")
  .setMaster("local[1]")
val sc = NNContext.initNNContext(conf)
val data = new Array[Sample[Float]](4)
var i = 0
while (i < data.length) {
  val input = Tensor[Float](2).fill(1.0f)
  val label = Tensor[Float](2).fill(1.0f)
  data(i) = Sample(input, label)
  i += 1
}
val model = Sequential[Float]().add(Linear(2, 2)).add(LogSoftMax())
val dataSet = sc.parallelize(data, 4)

val result = model.evaluate(dataSet, Array(new AUC[Float](20).asInstanceOf[ValidationMethod[Float]]))
```

**Python:**
```python
validation = AUC(20)
```
example
```
from zoo.common.nncontext import *
from bigdl.nn.layer import *
from zoo.pipeline.api.keras.metrics import AUC

sc = init_nncontext()

data_len = 4
batch_size = 8
FEATURES_DIM = 2

def gen_rand_sample():
    features = np.random.uniform(0, 1, (FEATURES_DIM))
    label = np.ones(FEATURES_DIM)
    return Sample.from_ndarray(features, label)

trainingData = sc.parallelize(range(0, data_len)).map(
    lambda i: gen_rand_sample())

model = Sequential()
model.add(Linear(2, 2)).add(LogSoftMax())
test_results = model.evaluate(trainingData, batch_size, [AUC(20)])

```
