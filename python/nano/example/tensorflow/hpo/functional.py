
import tensorflow as tf
from tensorflow import keras
from bigdl.nano.tf.keras.layers import Dense
from bigdl.nano.automl.tf.keras import Model
import bigdl.nano.automl.hpo.space as space


inputs = keras.Input(shape=(784,))

# Just for demonstration purposes.
img_inputs = keras.Input(shape=(32, 32, 3))

x = Dense(units=space.Categorical(8,16), activation="relu")(inputs)
x = Dense(units=space.Categorical(32,64), activation="relu")(x)
outputs = Dense(units=10)(x)

from bigdl.nano.automl.hpo.callgraph import plot_callgraph
plot_callgraph(outputs._callgraph)

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
