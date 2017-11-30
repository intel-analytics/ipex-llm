## Load Caffe model

**Scala:**

```scala
Module.loadCaffeModel(defPath, modelPath)
```
**Python:**
```python
Model.load_caffe_model(defPath, modelPath)
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Module.loadCaffeModel("/tmp/deploy.prototxt", "/tmp/caffe.caffemodel")
```

**Python example:**

``` python
from bigdl.nn.layer import *
model = Model.load_caffe_model("/tmp/deploy.prototxt", "/tmp/caffe.caffemodel")
```

## Load weight from Caffe into pre-defined Model

**Scala:**

```scala
Module.loadCaffe(model, defPath, modelPath, match_all = true)
```
**Python:**
```python
Model.load_caffe(model, defPath, modelPath, match_all = True)
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Sequential().add(Linear(3, 4))
val loadedModel = Module.loadCaffe(model, "/tmp/deploy.prototxt", "/tmp/caffe.caffemodel", true)
```

**Python example:**

``` python
from bigdl.nn.layer import *
model = Sequential().add(Linear(3, 4))
loadedModel = Model.load_caffe(model, "/tmp/deploy.prototxt", "/tmp/caffe.caffemodel", True)
```

## Save BigDL model as Caffe model

**Scala:**

```scala
bigdlModel.saveCaffe(prototxtPath, modelPath, useV2 = true, overwrite = false)
```
**Python:**
```python
bigdl_model.save_caffe(prototxt_path, model_path, use_v2 = True, overwrite = False)
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
val linear = Linear(3, 4)
val model = Graph(linear.inputs(), linear.inputs())
model.saveCaffe("/tmp/linear.prototxt", "/tmp/linear.caffemodel", true, true)
```

**Python example:**

``` python
from bigdl.nn.layer import *
linear = Linear(3, 4)
model = Graph(linear.inputs(), linear.inputs())
model.save_caffe(model, "/tmp/linear.prototxt", "/tmp/linear.caffemodel", True, True)
```