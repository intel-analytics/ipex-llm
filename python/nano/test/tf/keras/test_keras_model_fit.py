from bigdl.nano.tf.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unittest


def test_fit_batch_size():

    # Create Model
    inputs = tf.keras.Input(shape=(784,), name="digits")
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Prepare data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # Complie Model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # "Fit model on training data"
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
        perf_tune="batch_size"
    )
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("predictions shape:", predictions.shape)


if __name__ == '__main__':
    test_fit_batch_size()
