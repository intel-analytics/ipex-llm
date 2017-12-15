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


def get_mnist(sc, data_type="train", location="/tmp/mnist"):
    """
    Download or load MNIST dataset.
    Normalize and transform input data into an RDD of Sample
    """
    from bigdl.dataset import mnist
    from bigdl.dataset.transformer import normalizer
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = images.reshape(images.shape[0], 1, 28, 28)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1)  # Target start from 1 in BigDL
    record = images.zip(labels).map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),
                                    rec_tuple[1])) \
                               .map(lambda t: Sample.from_ndarray(t[0], t[1]))
    return record


def build_keras_model():
    """
    Define a convnet model in Keras
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D

    keras_model = Sequential()
    keras_model.add(Convolution2D(32, 3, 3, border_mode='valid',
                                  input_shape=(1, 28, 28)))
    keras_model.add(Activation('relu'))
    keras_model.add(Convolution2D(32, 3, 3))
    keras_model.add(Activation('relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Dropout(0.25))

    keras_model.add(Flatten())
    keras_model.add(Dense(128))
    keras_model.add(Activation('relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(10))
    keras_model.add(Activation('softmax'))

    return keras_model


def save_keras_model(keras_model, path):
    """
    Save a Keras model to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)


if __name__ == "__main__":
    keras_model = build_keras_model()
    def_path = "/tmp/lenet.json"
    save_keras_model(keras_model, def_path)

    from bigdl.util.common import *
    from bigdl.nn.layer import *
    from bigdl.optim.optimizer import *
    from bigdl.nn.criterion import *

    # Load the JSON file to a BigDL model
    bigdl_model = Model.load_keras(def_path=def_path)

    sc = get_spark_context(conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    train_data = get_mnist(sc, "train")
    test_data = get_mnist(sc, "test")

    optimizer = Optimizer(
        model=bigdl_model,
        training_rdd=train_data,
        criterion=ClassNLLCriterion(logProbAsInput=False),
        optim_method=Adadelta(),
        end_trigger=MaxEpoch(12),
        batch_size=128)
    optimizer.set_validation(
        batch_size=128,
        val_rdd=test_data,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )
    optimizer.optimize()
