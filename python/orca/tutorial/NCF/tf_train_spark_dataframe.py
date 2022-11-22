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
from process_spark_dataframe import data_process

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

# Step 1: Init Orca Context
init_orca_context(memory='4g')

# Step 2: Read and process data using Spark DataFrame
data_dir = './ml-1m'  # path to ml-1m
cat_feature = ['zipcode', 'gender', 'occupation', 'category']
num_feature = ['age']
total_features = ['user', 'item'] + cat_feature + num_feature

train_df, val_df, embedding_in_dim, user_num, item_num = data_process(
    data_dir, cat_feature, num_feature, neg_scale=4)

# Step 3: Define the NCF model
config = dict(
    factor_num=16,
    lr=1e-3,
    item_num=item_num,
    user_num=user_num,
    dropout=0.5,
    embedding_in_dim=embedding_in_dim,
    num_feature_dim=[1],
    embedding_out_dim=8
)


def model_creator(config):
    model = ncf_model(factor_num=config['factor_num'],
                      user_num=config['user_num'],
                      item_num=config['item_num'],
                      dropout=config['dropout'],
                      lr=config['lr'],
                      cat_features_in_dim=config['embedding_in_dim'],
                      cat_features_out_dim=config['embedding_out_dim'],
                      num_feature_dim=config['num_feature_dim'])
    return model


# Step 4: Distributed training with Orca keras Estimator
backend = 'spark'  # 'ray' of 'spark'
est = Estimator.from_keras(model_creator=model_creator,
                           config=config,
                           backend=backend)

batch_size = 256
train_steps = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(val_df.count() / batch_size)

est.fit(train_df,
        epochs=10,
        batch_size=batch_size,
        feature_cols=total_features,
        label_cols=['label'],
        steps_per_epoch=train_steps)

# Step 5: Distributed evaluation of the trained model
result = est.evaluate(val_df,
                      feature_cols=total_features,
                      label_cols=['label'],
                      batch_size=batch_size,
                      num_steps=val_steps)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])

# Step 6: Save the trained Tensorflow model
est.save("NCF_model")

# Step 7: Stop Orca Context when program finishes
stop_orca_context()
