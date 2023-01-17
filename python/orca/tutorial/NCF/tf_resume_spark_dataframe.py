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
import pickle

import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator

from process_spark_dataframe import get_feature_cols

# Step 1: Init Orca Context
init_orca_context(cluster_mode="local")
spark = OrcaContext.get_spark_session()

# Step 2: Read and process data using Spark DataFrame
train_df = spark.read.parquet("./train_processed.parquet")
test_df = spark.read.parquet("./test_processed.parquet")
feature_cols = get_feature_cols()
label_cols = ["label"]

# Step 3: Distributed training with Orca TF2 Estimator and load the model weight
backend = 'spark'  # 'ray' or 'spark'
est = Estimator.from_keras(backend=backend)
est.load('NCF_model_ray')

batch_size = 10240
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./log")]


def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * 0.1


with open('lr_callback.pkl', 'rb') as f:
    lr_callback = pickle.load(f)
callbacks.append(lr_callback)

est.fit(train_df,
        epochs=6,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_cols=label_cols,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        # initial_epoch=3,
        verbose=True)

# Step 4: Distributed evaluation of the trained model
result = est.evaluate(test_df,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      batch_size=batch_size,
                      num_steps=val_steps)
print('Evaluation results:')
for r in result:
    print("{}: {}".format(r, result[r]))

# Step 5: Save the trained TensorFlow model
est.save("NCF_resume_model")

# Step 6: Stop Orca Context when program finishes
stop_orca_context()
