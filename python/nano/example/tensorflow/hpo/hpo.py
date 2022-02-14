import tensorflow as tf

from bigdl.nano.automl.tf.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
import bigdl.nano.automl.hpo as hpo
import bigdl.nano.automl.hpo.space as space
from tensorflow.keras.layers import Flatten


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000

CLASSES = 10


(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
img_x, img_y = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
x_valid = x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
y_train = y_train[:N_TRAIN_EXAMPLES]
y_valid = y_valid[:N_VALID_EXAMPLES]
input_shape = (img_x, img_y, 1)

# decorate the layer class to accept automl.hpo.space as input argument
@hpo.obj()
class Conv2D(tf.keras.layers.Conv2D):
    pass

print(input_shape)
# define the model as usual. You can now use search space specificaions for Dense arguments
model = Sequential()

model.add(Conv2D(
            filters=space.Categorical(32, 64),
            kernel_size=space.Categorical(3, 5),
            strides=space.Categorical(1, 2),
            activation=space.Categorical("relu", "linear"),
            input_shape=input_shape))
model.add(Flatten())
model.add(Dense(CLASSES, activation="softmax"))

model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(learning_rate=0.0001),
        metrics=["accuracy"]
    )

model.search(
    n_trails = 2,
    target_metric='accuracy',
    direction="maximize",
    x = x_train,
    y = y_train,
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
        batch_size=128,
        epochs=2,
        verbose=False
    )
print(model.summary())

score = model.evaluate(x_valid, y_valid, verbose=0)

print("The final score is on validation data is",score[1])
