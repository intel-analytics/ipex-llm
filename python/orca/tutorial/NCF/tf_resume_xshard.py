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

from utils import *
from process_xshards import get_feature_cols

from bigdl.orca.data import XShards
from bigdl.orca.learn.tf2 import Estimator


# Step 1: Init Orca Context
args = parse_args("TensorFlow NCF Training with Spark DataFrame")
init_orca(args, extra_python_lib="tf_model.py")


# Step 2: Read and process data using Xshards
train_data = XShards.load_pickle(os.path.join(args.data_dir, "train_processed_xshards"))
test_data = XShards.load_pickle(os.path.join(args.data_dir, "test_processed_xshards"))
feature_cols = get_feature_cols()
label_cols = ["label"]


# Step 3: Distributed training with Orca TF2 Estimator and load the model weight
backend = 'ray'  # 'ray' or 'spark'
est = Estimator.from_keras(backend=backend)
est.load(os.path.join(args.model_dir, "NCF_model"))

batch_size = 10240
train_steps = math.ceil(len(train_data) / batch_size)
val_steps = math.ceil(len(test_data) / batch_size)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_dir, "logs"))] \
    if args.tensorboard else []

if args.scheduler:
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    callbacks.append(lr_callback)

    with open(os.path.join(args.model_dir, 'lr_callback.pkl'), 'rb') as f:
        lr_callback = pickle.load(f)
    callbacks.append(lr_callback)

est.fit(train_data,
        epochs=5,
        batch_size=batch_size,
        feature_cols=feature_cols,
        validation_data=test_data,
        validation_steps=val_steps,
        label_cols=label_cols,
        steps_per_epoch=train_steps,
        callbacks=callbacks)


# Step 4: Save the trained TensorFlow model
est.save(os.path.join(args.model_dir, "NCF_resume_model"))


# Step 5: Stop Orca Context when program finishes
stop_orca_context()
