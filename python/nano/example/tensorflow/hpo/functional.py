from select import epoll
import numpy as np

from tensorflow import keras
from bigdl.nano.tf.keras import layers
from bigdl.nano.automl.tf.keras import Model
import bigdl.nano.automl.hpo.space as space

#from tensorflow.keras import layers
#from tensorflow.keras import Model

inputs = keras.Input(shape=(784,))

# Just for demonstration purposes.
img_inputs = keras.Input(shape=(32, 32, 3))

dense = layers.Dense(32, activation="relu")
x = dense(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(10)(x)

# print(outputs._callgraph.nodes)

model = Model(inputs=inputs, outputs=outputs, name="mnist_model")


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.search(
    n_trails=2,
    target_metric='accuracy',
    direction="maximize",
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=1,
    validation_split=0.2,
    verbose=False,
)

history = model.fit(x_train, y_train,
                    batch_size=128, epochs=1, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

print(model.summary())
