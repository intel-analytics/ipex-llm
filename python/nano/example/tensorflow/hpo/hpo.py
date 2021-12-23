#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# this file is adapted from the optuna example https://github.com/optuna/optuna-examples/blob/main/keras/keras_simple.py

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.optimizers import RMSprop

from bigdl.nano.tf.keras import Sequential
from bigdl.nano.tf.keras import Model

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000

CLASSES = 10

# model_initor takes only a trial argument
# use lambda to accomodate extra arguments as examples shown in main
def model_inits(trial, input_shape):
    inputs = Input(shape=input_shape)
    conv = Conv2D(
            filters=trial.suggest_categorical("filters", [32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            strides=trial.suggest_categorical("strides", [1, 2]),
            activation=trial.suggest_categorical("activation", ["relu", "linear"]),
        ) (inputs)
    flat = Flatten()(conv)
    outputs = Dense(CLASSES, activation="softmax")(flat)
    return {'inputs': inputs,
             'outputs': outputs,
            }


def sequential_inits(trial, input_shape):
    layers = [
        Conv2D(
            filters=trial.suggest_categorical("filters", [32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            strides=trial.suggest_categorical("strides", [1, 2]),
            activation=trial.suggest_categorical("activation", ["relu", "linear"]),
            input_shape=input_shape,
        ),
        Flatten(),
        Dense(CLASSES, activation="softmax")
    ]
    return {'layers': layers}

def model_compiler(model, trial):
    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(learning_rate=lr),
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":

    use_sequential = False

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    x_valid = x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_valid = y_valid[:N_VALID_EXAMPLES]
    input_shape = (img_x, img_y, 1)

    if use_sequential:
        # use lambda to accomodate extra arguments
        model_initor = lambda trial: sequential_inits(trial, input_shape)
        model = Sequential(
            model_initor=model_initor,
            model_compiler=model_compiler)
    else:
        model_initor = lambda trial: model_inits(trial, input_shape)
        model = Model(
            model_initor=model_initor,
            model_compiler=model_compiler)

    model.tune(
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

    model.end_tune()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=128,
        epochs=3,
        verbose=False
    )

    score = model.evaluate(x_valid, y_valid, verbose=0)

    print("The final score is on validation data is",score[1])





