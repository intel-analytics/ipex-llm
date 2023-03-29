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

from process_spark_dataframe import prepare_data
from tf_model import ncf_model
from utils import *

from bigdl.orca.learn.tf2 import Estimator


# Step 1: Init Orca Context
args = parse_args("TensorFlow NCF Training with Spark DataFrame")
init_orca(args.cluster_mode, extra_python_lib="process_spark_dataframe.py,tf_model.py,utils.py")


# Step 2: Read and process data using Spark DataFrame
train_df, test_df, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data(args.data_dir, args.dataset, neg_scale=4)


# Step 3: Define the NCF model
config = dict(
    user_num=user_num,
    item_num=item_num,
    factor_num=16,
    num_layers=3,
    dropout=0.5,
    lr=0.01,
    sparse_feats_input_dims=sparse_feats_input_dims,
    sparse_feats_embed_dims=8,
    num_dense_feats=num_dense_feats
)


def model_creator(config):
    model = ncf_model(user_num=config["user_num"],
                      item_num=config["item_num"],
                      num_layers=config["num_layers"],
                      factor_num=config["factor_num"],
                      dropout=config["dropout"],
                      lr=config["lr"],
                      sparse_feats_input_dims=config["sparse_feats_input_dims"],
                      sparse_feats_embed_dims=config["sparse_feats_embed_dims"],
                      num_dense_feats=config["num_dense_feats"])
    return model


# Step 4: Distributed training with Orca TF2 Estimator
est = Estimator.from_keras(model_creator=model_creator,
                           config=config,
                           backend=args.backend,
                           workers_per_node=args.workers_per_node)

batch_size = 10240
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.model_dir, "logs"))] \
    if args.tensorboard else []

if args.lr_scheduler:
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule_func, verbose=1)
    callbacks.append(lr_callback)

train_stats = est.fit(train_df,
                      epochs=2,
                      batch_size=batch_size,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      steps_per_epoch=train_steps,
                      validation_data=test_df,
                      validation_steps=val_steps,
                      callbacks=callbacks)
print("Train results:")
for k, v in train_stats.items():
    print("{}: {}".format(k, v))


# Step 5: Distributed evaluation of the trained model
eval_stats = est.evaluate(test_df,
                          feature_cols=feature_cols,
                          label_cols=label_cols,
                          batch_size=batch_size,
                          num_steps=val_steps)
print("Evaluation results:")
for k, v in eval_stats.items():
    print("{}: {}".format(k, v))


# Step 6: Save the trained TensorFlow model and processed data for resuming training or prediction
est.save(os.path.join(args.model_dir, "NCF_model.h5"))
save_model_config(config, args.model_dir, "config.json")
train_df.write.parquet(os.path.join(args.data_dir,
                                    "train_processed_dataframe.parquet"), mode="overwrite")
test_df.write.parquet(os.path.join(args.data_dir,
                                   "test_processed_dataframe.parquet"), mode="overwrite")


# Step 7: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
