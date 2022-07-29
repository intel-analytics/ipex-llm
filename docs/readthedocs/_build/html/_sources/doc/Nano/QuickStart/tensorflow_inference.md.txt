# BigDL-Nano TensorFlow Inference Overview
BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. Currently, performance accelerations are achieved by integrating extra runtimes as inference backend engines or using quantization methods on full-precision trained models to reduce computation during inference. Keras Model (`bigdl.nano.tf.keras.Model`) and Sequential (`bigdl.nano.tf.keras.Sequential`) provides the APIs for all optimizations you need for inference. 

For quantization, BigDL-Nano provides only post-training quantization in `Model.quantize()` for users to infer with models of 8-bit precision. Quantization-Aware Training is not available for now. Model conversion to 16-bit like BF16, and FP16 will be coming soon.

Before you go ahead with these APIs, you have to make sure BigDL-Nano is correctly installed for Tensorflow. If not, please follow [this](../Overview/nano.md) to set up your environment.

## Quantization
Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. BigDL-Nano provides `Model.quantize()` API for users to quickly obtain a quantized model with accuracy control by specifying a few arguments. `Sequential` has similar usage, so we will only show how to use an instance of `Model` to enable quantization pipeline. 

To use INC as your quantization engine, you can choose accelerator as None or 'onnxruntime'. Otherwise, accelerator='openvino' means using OpenVINO POT to do quantization.

By default, `Model.quantize()` doesn't search the tuning space and returns the fully-quantized model without considering the accuracy drop. If you need to search quantization tuning space for a model with accuracy control, you'll have to specify a few arguments to define the tuning space. More instructions in [Quantization with Accuracy control](#quantization-with-accuracy-control)

### Quantization using Intel Neural Compressor
By default, Intel Neural Compressor is not installed with BigDL-Nano. So if you determine to use it as your quantization backend, you'll need to install it first:
```shell
# We have tested on neural-compressor>=1.8.1,<=1.11.0
pip install 'neural-compressor>=1.8.1,<=1.11.0'
```
**Quantization without extra accelerator**  
Without extra accelerators, `Model.quantize()` returns a Keras module with desired precision and accuracy. Taking MobileNetV2 as an example, you can add quantization as below:
```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from bigdl.nano.tf.keras import Model

# step 1: create your model
model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
model = Model(inputs=model.inputs, outputs=model.outputs)

# step 2: prepare your data and dataloader
train_examples = np.random.random((100, 40, 40, 3))
train_labels = np.random.randint(0, 10, size=(100,))
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

# (Optional) step 3: Something else, like training ...
model.fit(train_dataset)

# step 4: execute quantization
q_model = model.quantize(calib_dataset=train_dataset)

# run simple prediction
y_hat = q_model(train_examples)

# evaluate, predict also support acceleration
q_model.evaluate(train_dataset)
q_model.predict(train_dataset)
```
This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop.

To use quantization, you must use functional API to create a keras model. This is a known limitation
of INC.

### Quantization with Accuracy Control
A set of arguments that helps to tune the results for both INC and POT quantization:

- `calib_dataset`: A tf.data.Dataset object for calibration. Required for static quantization. It's also used as a validation dataloader.
- `metric`:  A `tensorflow.keras.metrics.Metric` object for evaluation.

- `accuracy_criterion`: A dictionary to specify the acceptable accuracy drop, e.g. `{'relative': 0.01, 'higher_is_better': True}`

    - `relative` / `absolute`: Drop type, the accuracy drop should be relative or absolute to baseline
    - `higher_is_better`: Indicate if a larger value of metric means better accuracy
- `max_trials`: Maximum trails on the search, if the algorithm can't find a satisfying model, it will exit and raise the error.
- `batch`: Specify the batch size of the dataloader. This will only take effect on evaluation. If it's not set, then we use `batch=1` for evaluation.

**Accuracy Control with INC**
There are a few arguments that require only by INC.
- `tuning_strategy`(optional): it specifies the algorithm to search the tuning space. In most cases, you don't need to change it.
- `timeout`: Timeout of your tuning. Defaults 0 means endless time for tuning.
- `inputs`:      A list of input names. Default: None, automatically get names from the graph.
- `outputs`:     A list of output names. Default: None, automatically get names from the graph.
Here is an example to use INC with accuracy control as below. It will search for a model within 1% accuracy drop with 10 trials.
```python
from torchmetrics.classification import Accuracy

q_model = model.quantize(precision='int8',
                         accelerator=None,
                         calib_dataset= train_dataset,
                         metric=Accuracy(),
                         accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                         approach='static',
                         tuning_strategy='bayesian',
                         timeout=0,
                         max_trials=10,
                         )

# run simple prediction
y_hat = q_model(train_examples)

# evaluate, predict also support acceleration
q_model.evaluate(train_dataset)
q_model.predict(train_dataset)
```