TFPredictor takes a list of TensorFlow tensors as the model outputs and feed all the elements in
 TFDatasets to produce those outputs and returns a Spark RDD with each of its elements representing the
 model prediction for the corresponding input elements.

 __Note__: This feature currently requires __tensorflow 1.10__ and OS is one of the following 64-bit systems.
 __Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.

 To run on other system may require you to manually compile the TensorFlow source code. Instructions can
 be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

**Python**
```python
logist = ...
predictor = TFPredictor.from_outputs(sess, [logits])
predictions_rdd = predictor.predict()
```

For Keras model:
```python
model = Model(inputs=..., outputs=...)
model.load_weights("/tmp/mnist_keras.h5")
predictor = TFPredictor.from_keras(model, dataset)
predictions_rdd = predictor.predict()
```
