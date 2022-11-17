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

from tf_model import ncf_model
from process_spark_dataframe import read_data, generate_neg_sample

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

# Step 1: Init Orca Context
init_orca_context(memory='4g')

# Step 2: Read and process data using Spark DataFrame
data_dir = './ml-1m'  # path to ml-1m
df = read_data(data_dir)
user_num = df.agg({'user': "max"}).collect()[0]["max(user)"] + 1
item_num = df.agg({'item': "max"}).collect()[0]["max(item)"] + 1
df = generate_neg_sample(df, item_num)
train_df, val_df = df.randomSplit([0.8, 0.2], seed=100)

# Step 3: Define the NCF model
config = dict(
    embedding_size=16,
    lr=1e-3,
    item_num=item_num,
    user_num=user_num,
    dropout=0.5,
)
batch_size = 256


def model_creator(config):
    model = ncf_model(embedding_size=config['embedding_size'],
                      user_num=config['user_num'],
                      item_num=config['item_num'],
                      dropout=config['dropout'],
                      lr=config['lr'])
    return model


# Step 4: Distributed training with Orca keras Estimator
backend = 'spark'  # 'ray' of 'spark'
est = Estimator.from_keras(model_creator=model_creator,
                           config=config,
                           backend=backend)
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(val_df.count() / batch_size)

est.fit(train_df,
        epochs=10,
        batch_size=batch_size,
        feature_cols=['user', 'item'],
        label_cols=['label'],
        steps_per_epoch=train_steps,
        validation_data=val_df,
        validation_steps=val_steps)

# Step 5: Distributed evaluation of the trained model
stats = est.evaluate(val_df,
                     feature_cols=['user', 'item'],
                     label_cols=['label'],
                     batch_size=batch_size,
                     num_steps=val_steps)
print("Evaluation results:", stats)

# Step 6: Save the trained Tensorflow model
est.save("NCF_model")

# Step 7: Stop Orca Context when program finishes
stop_orca_context()
