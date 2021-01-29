#
# Copyright 2018 Analytics Zoo Authors.
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
# ==============================================================================
#
# Copyright 2019 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This example shows how to classify cats vs dogs images
# by using transfer learning from a pre-trained network on Orca,
# and is adapted from
# https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/images/transfer_learning.ipynb

import os
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local or yarn.')
args = parser.parse_args()
cluster_mode = args.cluster_mode

dataset_dir = "datasets"
zip_file = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
    fname="cats_and_dogs_filtered.zip", extract=True, cache_dir="./")
base_dir, _ = os.path.splitext(zip_file)

if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4, memory="3g")
elif cluster_mode == "yarn":
    additional = "datasets/cats_and_dogs_filtered.zip#" + dataset_dir
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="3g",
                      additional_archive=additional)
else:
    print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
          + cluster_mode)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
print('Total training cat images:', len(os.listdir(train_cats_dir)))

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
print('Total training dog images:', len(os.listdir(train_dogs_dir)))

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

image_size = 160  # All images will be resized to 160x160
batch_size = 32


# Rescale all images by 1./255 and apply image augmentation
def _parse_function(img, label):
    image_resized = tf.image.resize(img, [image_size, image_size])
    return tf.cast(image_resized, dtype=tf.float32) / 255., tf.cast(label, dtype=tf.uint8)


builder = tfds.ImageFolder(base_dir)
ds = builder.as_dataset(shuffle_files=True, as_supervised=True)
train_dataset = ds['train'].map(_parse_function)

validation_dataset = ds['validation'].map(_parse_function)

IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
base_model.summary()

model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

epochs = 3
est = Estimator.from_keras(keras_model=model)
est.fit(train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_dataset
        )
result = est.evaluate(validation_dataset)
print(result)

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
print(len(model.trainable_variables))

est = Estimator.from_keras(keras_model=model)
est.fit(data=train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_dataset
        )
print("==>unfreeze")
result = est.evaluate(validation_dataset)
print(result)
stop_orca_context()
