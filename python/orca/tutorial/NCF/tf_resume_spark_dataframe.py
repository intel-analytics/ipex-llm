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

import tensorflow as tf

from utils import *
from process_spark_dataframe import get_feature_cols, get_label_cols

from bigdl.orca.learn.tf2 import Estimator


# Step 1: Init Orca Context
args = parse_args("TensorFlow NCF Resume Training with Spark DataFrame")
init_orca(args)
spark = OrcaContext.get_spark_session()


# Step 2: Read and process data using Spark DataFrame
train_df = spark.read.parquet(os.path.join(args.data_dir,
                                           "train_processed_dataframe.parquet"))
test_df = spark.read.parquet(os.path.join(args.data_dir,
                                          "test_processed_dataframe.parquet"))
feature_cols, label_cols = get_feature_cols(), get_label_cols()


# Step 3: Distributed training with Orca TF2 Estimator and load the model weight
est = Estimator.from_keras(backend=args.backend)
est.load(os.path.join(args.model_dir, "NCF_model"))

batch_size = 10240
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_dir, "logs"))] \
    if args.tensorboard else []

if args.lr_scheduler:
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    callbacks.append(lr_callback)

train_stats = est.fit(train_df,
                      initial_epoch=2,
                      epochs=4,
                      batch_size=batch_size,
                      steps_per_epoch=train_steps,
                      validation_data=test_df,
                      validation_steps=val_steps,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      callbacks=callbacks)
print("Train results:")
for k, v in train_stats.items():
    print("{}: {}".format(k, v))


# Step 4: Save the trained TensorFlow model
est.save(os.path.join(args.model_dir, "NCF_resume_model"))


# Step 5: Stop Orca Context when program finishes
stop_orca_context()
