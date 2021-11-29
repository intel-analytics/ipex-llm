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

# Related url: https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/images/transfer_learning.ipynb
# Categorize image to cat or dog   
import os
import tensorflow.compat.v1 as tf
from tensorflow import keras

# Obtain data from url:"https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                                   fname="cats_and_dogs_filtered.zip", extract=True)

# Find the directory of validation set
base_dir, _ = os.path.splitext(zip_file)
test_dir = os.path.join(base_dir, 'validation')

# Set images size to 160x160x3
image_size = 160

# Rescale all images by 1./255 and apply image augmentation
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow images using generator to the test_generator
test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(image_size, image_size),
                batch_size=1,
                class_mode='binary')

# Convert the next data of ImageDataGenerator to ndarray
def convert_to_ndarray(ImageGenerator):
    return ImageGenerator.next()[0]

# Load model from its path
model=tf.keras.models.load_model("path/to/model")

# Convert each image in test_generator to ndarray and predict with model
max_length=test_generator.__len__()
for i in range(max_length): # number of image to predict can be altered
    test_input=convert_to_ndarray(test_generator)
    prediction=model.predict(test_input)

