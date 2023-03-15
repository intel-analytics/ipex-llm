# TensorFlow Inference

BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. Currently, performance accelerations are achieved by integrating extra runtimes as inference backend engines or using quantization methods on full-precision trained models to reduce computation during inference. InferenceOptimizer(`bigdl.nano.tf.keras.InferenceOptimizer`) provides the APIs for all optimizations that you need for inference.


## Automatically Choose the Best Optimization

We recommend you to use `InferenceOptimizer.optimize` to compare different optimization methods and choose the best one.

Taking MobileNetV2 as an example, you can use runtime acceleration as below:

```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from bigdl.nano.tf.keras import InferenceOptimizer

# step 1: create your model
model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)

# step 2: prepare your data and dataset
train_examples = np.random.random((100, 40, 40, 3))
train_labels = np.random.randint(0, 10, size=(100,))
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

# (Optional) step 3: Something else, like training ...
model.fit(train_dataset)

# step 4: try all supproted optimizations
opt = InferenceOptimizer()
opt.optimize(model, x=train_dataset)

# get the best optimization
best_model, _option = opt.get_best_model()

# use the quantized model as before
y_hat = best_model(train_examples)
best_model.predict(train_dataset)
```

`InferenceOptimizer.optimize` will try all supported optimizations and choose the best one. e.g. the output may like this

```
==========================Optimization Results==========================
 -------------------------------- ---------------------- --------------
|             method             |        status        | latency(ms)  |
 -------------------------------- ---------------------- --------------
|            original            |      successful      |    82.109    |
|              int8              |      successful      |    4.398     |
|         openvino_fp32          |      successful      |    3.847     |
|         openvino_int8          |      successful      |    2.177     |
|        onnxruntime_fp32        |      successful      |     3.28     |
|    onnxruntime_int8_qlinear    |      successful      |    3.071     |
|    onnxruntime_int8_integer    |   fail to convert    |     None     |
 -------------------------------- ---------------------- --------------
```

```eval_rst
.. tip::
    It also uses parameter ``x`` and ``y`` to receive calibration data like ``InferenceOptimizer.quantize``.

    There are some other useful parameters

    - ``includes``: A str list. If set, ``optimize`` will only try optimizations in this parameter.
    - ``excludes``: A str list. If set, ``optimize`` will try all optimizations (or optimizations specified by ``includes``) except for those in this parameter.

    See its `API document <../../PythonAPI/Nano/tensorflow.html#bigdl.nano.tf.keras.InferenceOptimizer.optimize>`_ for more advanced usage.
```

Before you go ahead with these APIs, you have to make sure BigDL-Nano is correctly installed for TensorFlow. If not, please follow [this](./install.md) to set up your environment.

```eval_rst
.. note::
    You can install all required dependencies by

    ::

        pip install bigdl-nano[tensorflow,inference]

    This will install all dependencies required by BigDL-Nano TensorFlow inference.

    Or if you just want to use one of supported optimizations:

    - INC (Intel Neural Compressor): ``pip install neural-compressor``
    - OpenVINO: ``pip install openvino-dev``
    - ONNXRuntime: ``pip install onnx onnxruntime onnxruntime-extensions tf2onnx neural-compressor``

    We recommand installing all dependencies by ``pip install bigdl-nano[tensorflow,inference]``, because you may run into version issues if you install dependencies manually.
```

## Manually Chose Optimizations

### Runtime Acceleration

For runtime acceleration, BigDL-Nano has enabled two kinds of runtime (OpenVINO and ONNXRuntime) for users in `InferenceOptimizer.trace()`.

```eval_rst
.. warning::
    ``model.trace`` will be deprecated in future release.

    Please use ``bigdl.nano.tf.keras.InferenceOptimizer.trace`` instead.
```

All available runtime accelerations are integrated in `InferenceOptimizer.trace(accelerator='onnxruntime'/'openvino')` with different accelerator values.

Taking the example in [Automatically Choose the Best Optimization](#automatically-choose-the-best-optimization), you can use runtime acceleration as following:

```python
# execute quantization using `OpenVINO` acceleration
traced_model = InferenceOptimizer.trace(model, accelerator="openvino")
# execute quantization using `ONNXRuntime` acceleration
traced_model = InferenceOptimizer.trace(model, accelerator="onnxruntime")

# run simple prediction
y_hat = traced_model(train_examples)

# predict also support acceleration
traced_model.predict(train_dataset)
```

### Quantization

Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. BigDL-Nano provides `InferenceOptimizer.quantize()` API for users to quickly obtain a quantized model with accuracy control by specifying a few arguments.

BigDL-Nano currently provides only post-training quantization in `InferenceOptimizer.quantize()` for users to infer with models of 8-bit precision. Quantization-Aware Training is not available for now. Model conversion to 16-bit like BF16 and FP16 is coming soon.

```eval_rst
.. warning::
    ``model.quantize`` will be deprecated in future release.

    Please use ``bigdl.nano.tf.keras.InferenceOptimizer.quantize`` instead.
```

To use INC as your quantization engine, you can choose `accelerator=None/'onnxruntime'`. Otherwise, `accelerator='openvino'` means using OpenVINO POT (Post-training Optimization) to do quantization.

#### Quantization without Accuracy Control

Taking the example in [Runtime Acceleration](#runtime-acceleration), you can use quantization as following:

```python
# use Intel Neural Compressor quantization
q_model = InferenceOptimizer.quantize(model, x=train_dataset)

# or use ONNXRuntime quantization
q_model = InferenceOptimizer.quantize(model, x=train_dataset, accelerator="onnxruntime")

# or use OpenVINO quantization
q_model = InferenceOptimizer.quantize(model, x=train_dataset, accelerator="openvino")

# you can also use features and labels instead of dataset for quantization
q_model = InferenceOptimizer.quantize(model, x=train_examples, y=train_labels)

# you can use quantized model as before
y_hat = q_model(train_examples)
q_model.predict(train_dataset)
```

This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop.

```eval_rst
.. note::
    Now BigDL-Nano only support static quantization, which needs training data to do calibration. Parameter `x` and `y` are used to receive calibration data.

    - ``x``: Input data which is used for training. It could be

      - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
      - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
      - An unbatched ``tf.data.Dataset``. Should return tuples of (inputs, targets). (In this case there is no need to pass the parameter ``y``)

    - ``y``: Target data. Like the input data ``x``, it could be either Numpy array(s) or TensorFlow tensor(s). Its length should be consistent with ``x``. If ``x`` is a ``Dataset``, ``y`` will be ignored (since targets will be obtained from ``x``)
```

#### Quantization with Accuracy Control

By default, `InferenceOptimizer.quantize()` doesn't search the tuning space and returns the fully-quantized model without considering the accuracy drop. If you need to search quantization tuning space for a model with accuracy control, you may need to specify a few parameters.

Following parameters can help you tune the results for both INC and POT quantization:

- `metric`:  A `tensorflow.keras.metrics.Metric` object for evaluation.
- `accuracy_criterion`: A dictionary to specify the acceptable accuracy drop, e.g. `{'relative': 0.01, 'higher_is_better': True}`

    - `relative` / `absolute`: Drop type, the accuracy drop should be relative or absolute to baseline
    - `higher_is_better`: Indicate if a larger value of metric means better accuracy

- `max_trials`: Maximum trails on the search, if the algorithm can't find a satisfying model, it will exit and raise the error.
- `batch`: Specify the batch size of the dataset. This will only take effect on evaluation. If it's not set, then we use `batch=1` for evaluation.

**Accuracy Control with INC**

There are a few arguments that only take effect when using INC.
- `tuning_strategy` (optional): it specifies the algorithm to search the tuning space. In most cases, you don't need to change it.
- `timeout`: Timeout of your tuning. Default: 0, means endless time for tuning.
- `inputs`:      A list of input names. Default: None, automatically get names from the graph.
- `outputs`:     A list of output names. Default: None, automatically get names from the graph.
Here is an example to use INC with accuracy control as below. It will search for a model within 1% accuracy drop with 10 trials.

Here is an example to use INC with accuracy control as below. It will search for a model within 1% accuracy drop with 10 trials.
```python
from torchmetrics.classification import MulticlassAccuracy

q_model = InferenceOptimizer.quantize(model,
                                      x=train_dataset,
                                      accelerator=None,
                                      metric=MulticlassAccuracy(num_classes=10),
                                      accuracy_criterion={'relative': 0.01,
                                                          'higher_is_better': True},
                                      approach='static',
                                      tuning_strategy='bayesian',
                                      timeout=0,
                                      max_trials=10)

# use the quantized model as before
y_hat = q_model(train_examples)
q_model.predict(train_dataset)
```