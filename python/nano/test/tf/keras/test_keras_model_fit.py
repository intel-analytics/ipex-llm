import unittest
from bigdl.nano.tf.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



class ModelTestCase(unittest.TestCase):

    def test_fit_batch_size(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = Model([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        predictions = model(x_train[:1]).numpy()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(y_train[:1], predictions).numpy()
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        predictions = model(x_train[:1]).numpy()
        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test, verbose=2)
        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])
        probability_model(x_test[:1]).numpy()


if __name__ == '__main__':
    unittest.main()
