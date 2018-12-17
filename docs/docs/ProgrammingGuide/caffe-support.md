If you have a pretrained caffe model(model definition prototxt and model binary file), you can load it as BigDL model.
You can also convert a BigDL model to caffe model.

## **Load Caffe Model**

Assume you have a ```caffe.prototxt``` and ```caffe.model```,
you can load it into BigDL by calling ```Module.loadCaffeModel``` (scala) or ```Model.load_caffe_model``` (python).

* Scala Example

```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
val model = Module.loadCaffeModel(caffe.prototxt, caffe.model)
```

* Python Example
```python
model = Model.load_caffe_model(caffe.prototxt, caffe.model)
```

## **Load Caffe Model Weights to Predefined BigDL Model**
 
If you have a predefined BigDL model, and want to load caffe model weights into BigDl model

* Scala Example
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
val model = Module.loadCaffe(bigdlModel, caffe.prototxt, caffe.model, matchAll = true)
```

* Python Example
```python
model = Model.load_caffe(bigdlModel, caffe.prototxt, caffe.model, match_all=True)
```

Note that if ```matchAll/match_all = false```, then only layers with same name will be loaded, the rest will use initialized parameters.

## **Save BigDL Model to Caffe Model**
* Scala Example
```scala
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
bigdlModel.saveCaffe(prototxtPath, modelPath, useV2 = true, overwrite = false)
```

* Python Example
```python
bigdl_model.save_caffe(prototxt_path, model_path, use_v2 = True, overwrite = False)
```
In the above examples, if ```useV2/use_v2 = true```, it will convert to caffe V2 layer,
 otherwise, it will convert to caffe V1 layer.
If ```overwrite = true```, it will overwrite the existing files.

Note: only graph model can be saved to caffe model.

### **Limitation**
This functionality has been tested with some common models like AlexNet, Inception, Resnet which were created with standard Caffe layers, for those models with customized layers such as SSD, it is going to be supported in future work, but you can define your customized conversion method for your own layers.

### **Supported Layers**
Please check this [page](../APIGuide/caffe_layer_list.md)
