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

# Step 0: Import necessary libraries
import math

from model import xception_model
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
import tensorflow_datasets as tfds

# Step 1: Init Orca Context
init_orca_context(memory='4g')

# Step 2: Read and process data
def data_process():
    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        # Reserve 10% for validation and 10% for test
        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
        as_supervised=True,  # Include labels
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print(
        "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
    )
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

    return train_ds, validation_ds, test_ds
train_ds, validation_ds, test_ds = data_process()


def preprocess(x, y):
    x = tf.image.resize(x, (150, 150))
    return x, y


def train_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[0]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


def val_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[1]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


def test_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[2]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


# Step 3: Define model
config = dict(
    dropout=0.2
)


def model_creator(config):
    model = xception_model(config['dropout'])
    return model


# Step 4: Distributed training with Orca keras Estimator
backend = 'spark'  # 'ray' or 'spark'
est = Estimator.from_keras(model_creator=model_creator,
                           config=config,
                           backend=backend)

batch_size = 32
train_steps = math.ceil(tf.data.experimental.cardinality(train_ds) / batch_size)
val_steps = math.ceil(tf.data.experimental.cardinality(validation_ds) / batch_size)
test_steps = math.ceil(tf.data.experimental.cardinality(test_ds) / batch_size)
est.fit(data=train_data_creator,
        epochs=1,
        batch_size=batch_size,
        steps_per_epoch=train_steps,
        validation_data=val_data_creator,
        validation_steps=val_steps,
        data_config=config)

# Step 5: Distributed evaluation of the trained model
stats = est.evaluate(test_data_creator,
                     batch_size=batch_size,
                     num_steps=test_steps,
                     data_config=config)
print("Evaluation results:", stats)

# Step 6: Save the trained Tensorflow model
est.save("model")

# Step 7: Stop Orca Context when program finishes
stop_orca_context()
