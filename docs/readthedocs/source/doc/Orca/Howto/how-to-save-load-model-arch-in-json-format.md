# Save and Load Model Architecture in JSON Format 

A Keras model consists of multiple components:

The architecture, or configuration, a set of weights values, an optimizer and a set of losses and metrics(refer to [link](https://www.tensorflow.org/guide/keras/save_and_serialize#introduction))
The Keras API makes it possible to save all of these pieces to disk at once, or to only selectively save some of them. In this guide, we will illustrate how to save and load the architecture only, typically in JSON format.

The model's architecture specifies what layers the model contains, and how these layers are connected. If you have the configuration of a model, then the model can be created with a freshly initialized state for the weights and no compilation information.

**Note this only applies to models defined using the functional or Sequential apis not subclassed models.**

## Save the architecture
Obtain JSON string or config dictionary with `to_json()`. Then use python decorator to save in different file systems(such as local file system, HDFS and S3 file system) as follows:

```python
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(5, input_shape=(3,)),
                            tf.keras.layers.Softmax()])

from bigdl.orca.data.file import enable_multi_fs_save

@enable_multi_fs_save
def save_model_arch(model, path):
    config = model.to_json()
    with open(path, "wb") as f:
        f.write(config)
```

## Load the architecture

Load JSON string or config dictionary from different file systems(such as local file system, HDFS and S3 file system) with python decorator. Then use `model_from_json()` to parse a JSON model configuration string and returns a model instance.

```python
from bigdl.orca.data.file import enable_multi_fs_load

@enable_multi_fs_load
def load_model_arch(model, path):
    with open(path, "rb") as f:
        config = f.read()
    model = tf.keras.models.model_from_json(config)
```