# **Keras Support**

BigDL supports loading pre-defined Keras models and running the models in a distributed manner.

The Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/).

## **Loading a Keras model into BigDL**

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
You can also load saved weights in __HDF5__ file to the model.
See [here](https://faroit.github.io/keras-docs/1.2.2/getting-started/faq/#how-can-i-save-a-keras-model) on how to save the architecture and weights of a Keras model.

The API `load_keras` can be used directly to load the Keras model into BigDL.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(def_path, weights_path=None, by_name=False)
```
Parameters:

* `def_path` The JSON file path containing the keras model definition to be loaded.
* `weights_path`  The HDF5 file path containing the pre-trained keras model weights. Default to be `None` if you choose not to load weights. In this case, initialized weights will be used for the model.
* `by_name`  Whether to load the weights of layers by name. Use this option only when you do load the pre-trained weights. Default to be `False`, meaning that  weights are loaded based on the network's execution order topology. Otherwise, only those layers with the same name will be loaded with weights.

### **Limitation**
We have tested the model loading functionality with some standard [Keras applications](https://faroit.github.io/keras-docs/1.2.2/applications/) and [examples](https://github.com/fchollet/keras/tree/1.2.2/examples).

There still exist some arguments for Keras layers that are not supported in BigDL for now. We haven't supported self-defined layers, but one can still define your customized layer converter and weight converter method for new layers.

In our future work, we will continue add functionality and better support running Keras on BigDL.