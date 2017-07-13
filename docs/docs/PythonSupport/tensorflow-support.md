## Loading a tensorflow model from tensorflow model file

If you have a pre-trained tensorflow model saved in a ".pb" file. You load it
into BigDL using `Model.load_tensorflow` api.

**Python:**
```python
path = "your/path/to/tensorflow_model.pb"
inputs = ["your_input_node"]
outputs = ["your_output_node"]
model = Model.load_tensorflow(path, inputs, outputs, byte_order = "little_endian", bigdl_type="float")
```

## Build model using tensorflow and train with BigDL

You can construct your BigDL model directly from the input and output nodes of
tensorflow model. That is to say, you can use tensorflow to define
a model and use BigDL to run it.

**Python:**
```python
import tensorflow as tf
import numpy as np

tf.set_random_seed(1234)
input = tf.placeholder(tf.float32, [None, 5])
weight = tf.Variable(tf.random_uniform([5, 10]))
bias = tf.Variable(tf.random_uniform([10]))
middle = tf.nn.bias_add(tf.matmul(input, weight), bias)
output = tf.nn.tanh(middle)

tensor = np.random.rand(5, 5)
# construct BigDL model and get the result form 
bigdl_model = Model(input, output, model_type="tensorflow")
bigdl_result = bigdl_model.forward(tensor)

# get result from tensorflow and compare
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    tensorflow_result = sess.run(output, {input: tensor})

    print("Tensorflow forward result is " + str(tensorflow_result))
    print("BigDL forward result is " + str(bigdl_result))

    np.testing.assert_almost_equal(tensorflow_result, bigdl_result, 6)
    print("The results are almost equal in 6 decimals")
```
