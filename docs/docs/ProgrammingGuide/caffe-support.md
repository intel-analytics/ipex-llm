## **Loading a caffe model**

If you have a pretrained caffe model(model definition prototxt and model binary file), you can load it into BigDL model.

Assume you have a ```caffe.prototxt``` and ```caffe.model```,
you can load it into BigDL by calling ```Module.loadCaffeModel``` (scala) or ```Model.load_caffe_model``` (python).

### Load Caffe Model Scala Example
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
val model = Module.loadCaffeModel(caffe.prototxt, caffe.model)
```

### Load Caffe Model Python Example
```python
model = Model.load_caffe_model(caffe.prototxt, caffe.model)
```

If you have a predefined BigDL model, and want to load caffe model weights into BigDl model

### Load Caffe Model Weights to BigDL Model Scala Example
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
Module.loadCaffe(bigdlModel, caffe.prototxt, caffe.model, matchAll = true)
```
Note that if ```matchAll = false```, then only layers with same name will be loaded, the rest will use initialized parameters.

### Load Caffe Model Weights to BigDL Model Python Example
```python
model = Model.load_caffe_model(bigdlModel, caffe.prototxt, caffe.model, match_all=True)
```
