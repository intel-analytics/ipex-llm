## Overview

Analytics Zoo provides some useful utilities for transfer learning.

### Loading a pre-trained model

We can use the `Net` api to load a pre-trained model, including models saved by Analytics Zoo,
BigDL, Torch, Caffe and Tensorflow. Please refer to [Net API Guide](../APIGuide/PipelineAPI/net.md)


### Remove the last a few layers

When a model is loaded using `Net`, we can use the `newGraph(output)` api to define a Model with
the output specified by the parameter.

For example, 

In scala:
```scala
val inception = Net.loadBigDL[Float](inception_path)
      .newGraph(output = "pool5/drop_7x7_s1")

```

In python:
```python
full_model = Net.load_bigdl(model_path)
# create a new model by remove layers after pool5/drop_7x7_s1
model = full_model.new_graph(["pool5/drop_7x7_s1"])
```

The returning model's output layer is "pool5/drop_7x7_s1".

### Freeze some layers

In transfer learning, we often want to freeze some layers to prevent overfitting. In Analytics Zoo,
we can use the `freezeUpTo(endPoint)` api to do that.

For example,

In scala:
```scala
inception.freezeUpTo("pool4/3x3_s2") // freeze layer pool4/3x3_s2 and the layers before it
```

In python:
```python
# freeze layers from input to pool4/3x3_s2 inclusive
model.freeze_up_to(["pool4/3x3_s2"])
```

This will freeze all the layers from the input layer to "pool4/3x3_s2"

### Example

For a complete example, refer to the [scala transfer learning example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/finetune)
and [python transfer learning example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/nnframes/finetune)
