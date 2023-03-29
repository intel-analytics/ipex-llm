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

from process_spark_dataframe import get_feature_cols, get_label_cols
from utils import *

from bigdl.orca.learn.tf2 import Estimator


# Step 1: Init Orca Context
args = parse_args("TensorFlow NCF Resume Training with Spark DataFrame")
# TODO: fix spark backend for saving optimizer states
init_orca(args.cluster_mode, extra_python_lib="process_spark_dataframe.py,utils.py")
spark = OrcaContext.get_spark_session()


# Step 2: Read and process data using Spark DataFrame
train_df = spark.read.parquet(os.path.join(args.data_dir,
                                           "train_processed_dataframe.parquet"))
test_df = spark.read.parquet(os.path.join(args.data_dir,
                                          "test_processed_dataframe.parquet"))


# Step 3: Distributed training with Orca TF2 Estimator after loading the model
est = Estimator.from_keras(backend=args.backend,
                           workers_per_node=args.workers_per_node)
est.load(os.path.join(args.model_dir, "NCF_model.h5"))

batch_size = 10240
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_dir, "logs"))] \
    if args.tensorboard else []

if args.lr_scheduler:
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule_func, verbose=1)
    callbacks.append(lr_callback)

train_stats = est.fit(train_df,
                      initial_epoch=2,
                      epochs=4,
                      batch_size=batch_size,
                      feature_cols=get_feature_cols(),
                      label_cols=get_label_cols(),
                      steps_per_epoch=train_steps,
                      validation_data=test_df,
                      validation_steps=val_steps,
                      callbacks=callbacks)
print("Train results:")
for k, v in train_stats.items():
    print("{}: {}".format(k, v))


# Step 4: Save the trained TensorFlow model
est.save(os.path.join(args.model_dir, "NCF_resume_model.h5"))


# Step 5: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
