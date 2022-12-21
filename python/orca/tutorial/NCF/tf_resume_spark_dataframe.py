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

from tf_model import ncf_model
from process_spark_dataframe import prepare_data

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

# Step 1: Init Orca Context
init_orca_context(memory='4g')

# Step 2: Read and process data using Spark DataFrame
data_dir = './ml-1m'  # path to ml-1m

train_df, test_df, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
    feature_cols, label_cols = prepare_data(data_dir, neg_scale=4)

# Step 3: Distributed training with Orca keras Estimator and load the model weight
backend = 'ray'  # 'ray' or 'spark'

est = Estimator.from_keras()
est.load('NCF_model')

batch_size = 10240
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

est.fit(train_df,
        epochs=5,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_cols=label_cols,
        steps_per_epoch=train_steps,
        callbacks=[tf_callback])

# Step 4: Distributed evaluation of the trained model
result = est.evaluate(test_df,
                      feature_cols=feature_cols,
                      label_cols=label_cols,
                      batch_size=batch_size,
                      num_steps=val_steps,
                      )
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])

# Step 5: Save the trained Tensorflow model
est.save("NCF_model")

# Step 6: Stop Orca Context when program finishes
stop_orca_context()
