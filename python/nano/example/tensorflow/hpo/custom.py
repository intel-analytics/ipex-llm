
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop

import bigdl.nano.automl.hpo as hpo


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
CLASSES = 10

@hpo.model()
class MyModel(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides, activation):
        super().__init__()
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation)
        self.flat = Flatten()
        self.dense = Dense(CLASSES, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flat(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    x_valid = x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_valid = y_valid[:N_VALID_EXAMPLES]
    input_shape = (img_x, img_y, 1)


    model = MyModel(
            filters=hpo.space.Categorical(32, 64),
            kernel_size=hpo.space.Categorical(3, 5),
            strides=hpo.space.Categorical(1, 2),
            activation=hpo.space.Categorical("relu", "linear")
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    model.search(
        n_trails = 2,
        target_metric='accuracy',
        direction="maximize",
        x=x_train,
        y=y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=128,
        epochs=2,
        verbose=False,
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=120,
        epochs=2,
        verbose=True,
    )

print(model.summary())

score = model.evaluate(x_valid, y_valid, verbose=0)

print("The final score is on validation data is",score[1])