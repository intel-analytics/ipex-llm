## **Loading a tensorflow model from tensorflow model file**

If you have a pre-trained tensorflow model saved in a ".pb" file. You load it
into BigDL using `Model.load_tensorflow` api. 

For more information on how to generate
the ".pb" file, you can refer to [A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/extend/tool_developers/).
Specifically, you should generate a model definition file and a set of checkpoints, then use the [freeze_graph](https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/python/tools/freeze_graph.py)
script to freeze the graph definition and weights in checkpoints into a single file.

**Generate model definition file and checkpoints (in python)**
```python
import tensorflow as tf
xs = tf.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.zeros([1,10])+0.2)
b1 = tf.Variable(tf.zeros([10])+0.1)
Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs,W1), b1)
output = tf.nn.tanh(Wx_plus_b1, name="output")

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    checkpointpath = saver.save(sess, '/tmp/model/test.chkp')
    tf.train.write_graph(sess.graph, '/tmp/model', 'test.pbtxt')
```

**Freeze graph definition and checkpoints into a single ".pb" file (in shell)**
```shell
wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
python freeze_graph.py --input_graph /tmp/model/test.pbtxt --input_checkpoint /tmp/model/test.chkp --output_node_names=output --output_graph "/tmp/model/test.pb"
```

**Load tensorflow model (in python)**
```python
path = "/tmp/model/test.pb"
inputs = ["Placeholder"]
outputs = ["output"]
model = Model.load_tensorflow(path, inputs, outputs, byte_order = "little_endian", bigdl_type="float")
```
---
## **Build model using tensorflow and train with BigDL**

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
