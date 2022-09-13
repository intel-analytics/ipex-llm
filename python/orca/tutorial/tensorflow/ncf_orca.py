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

import time
import numpy as np
import tensorflow as tf

from bigdl.orca import OrcaContext, init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator


def model_creator(config):
    num_users = max_user_id
    num_items = max_item_id
    layers=[20, 10]
    include_mf=True
    mf_embed=20
    num_layer = len(layers)
    user_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="user_input")
    item_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="item_input")

    mlp_embed_user = tf.keras.layers.Embedding(input_dim=num_users + 1,
                                               output_dim=int(layers[0] / 2),
                                               input_length=1)(user_input)
    mlp_embed_item = tf.keras.layers.Embedding(input_dim=num_items + 1,
                                               output_dim=int(layers[0] / 2),
                                               input_length=1)(item_input)
    user_latent = tf.keras.layers.Flatten()(mlp_embed_user)
    item_latent = tf.keras.layers.Flatten()(mlp_embed_item)

    mlp_latent = tf.keras.layers.concatenate([user_latent, item_latent], axis=1)
    for idx in range(1, num_layer):
        layer = tf.keras.layers.Dense(layers[idx], activation="relu",
                      name="layer%d" % idx)
        mlp_latent = layer(mlp_latent)

    if include_mf:
        mf_embed_user = tf.keras.layers.Embedding(input_dim=num_users + 1,
                                                  output_dim=mf_embed,
                                                  input_length=1)(user_input)
        mf_embed_item = tf.keras.layers.Embedding(input_dim=num_items + 1,
                                                  output_dim=mf_embed,
                                                  input_length=1)(item_input)
        mf_user_flatten = tf.keras.layers.Flatten()(mf_embed_user)
        mf_item_flatten = tf.keras.layers.Flatten()(mf_embed_item)

        mf_latent = tf.keras.layers.multiply([mf_user_flatten, mf_item_flatten])
        concated_model = tf.keras.layers.concatenate([mlp_latent, mf_latent], axis=1)
        prediction = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(concated_model)
    else:
        prediction = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(mlp_latent)

    model = tf.keras.Model([user_input, item_input], prediction)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])


    return model

cluster_mode = "local"
# cluster_mode = "k8s"
if cluster_mode == "local":
    sc = init_orca_context(memory="20g")
elif cluster_mode == "standalone":
    sc = init_orca_context("standalone", master="spark://...", cores=8, num_nodes=4, memory="10g")
elif cluster_mode == "yarn":
    sc = init_orca_context("yarn", cores=8, num_nodes=4, memory="10g")
elif cluster_mode == "k8s":
    sc = init_orca_context(cluster_mode="k8s", cores=8, num_nodes=4, memory="20g",
                           master="k8s://https://172.16.0.200:6443",
                           container_image="10.239.45.10/arda/intelanalytics/bigdl-k8s-spark-3.1.2:0.14.0-SNAPSHOT",
                           conf={"spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data"})
elif cluster_mode == "spark-submit":  # To test k8s using spark-submit
    sc = init_orca_context(cluster_mode="spark-submit")


# data_path = "/home/kai/Downloads"
data_path = "./bigdl2.0/data"
data_type = "ml-1m"
# data_type = "ml-25m"
# Need spark3 to support delimiter with more than one character.
start = time.time()
spark= OrcaContext.get_spark_session()
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

schema = StructType(
    [
        StructField('_c0', IntegerType(), True),
        StructField('_c1', IntegerType(), True)
    ]
)
df0 = spark.read.csv("{}/{}/ratings.csv".format(data_path, data_type), sep="::",schema=schema,
                     header=False)
df0 = df0.withColumnRenamed('_c0','user').withColumnRenamed('_c1','item')
#print(int(df.describe("item").filter("summary = 'max'").select("item").first().asDict()['item']))
min_user_id = df0.agg({"user": "min"}).collect()[0]["min(user)"]
max_user_id = df0.agg({"user": "max"}).collect()[0]["max(user)"]
min_item_id = df0.agg({"item": "min"}).collect()[0]["min(item)"]
max_item_id = df0.agg({"item": "max"}).collect()[0]["max(item)"]
print(min_user_id, max_user_id, min_item_id, max_item_id)
from pyspark.sql import functions
df = df0.withColumn('label', functions.lit(1))


import random
import pandas as pd
def negative_sample(pandas_df):
    df_list = pandas_df.values.tolist()
    count_df = len(df_list) * 4
    mat = [[0] * (max_item_id + 1) for _ in range(max_user_id + 1)]
    for x in df_list:
        mat[x[0]][x[1]] = 1
    neg_list = []
    while count_df:
        neg_user = random.randint(min_user_id, max_user_id)
        neg_item = random.randint(min_item_id, max_item_id)
        neg_com = [neg_user, neg_item]
        if mat[neg_user][neg_item] == 0:
            neg_list.append(neg_com)
            count_df -= 1
            mat[neg_user][neg_item] = 1
    return pd.DataFrame(neg_list)
df_neg = df0.groupby().applyInPandas(negative_sample, schema=schema)
df_neg = df_neg.withColumn('label', functions.lit(0))


df.unionAll(df_neg)

train_df, test_df = df.randomSplit([0.8, 0.2],seed = 11)


batch_size=1280
epochs=2
model_dir='./'

# create an Estimator
est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1)

stats = est.fit(train_df,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                steps_per_epoch=800000 // batch_size,
                validation_data=test_df,
                validation_steps = 200000 // batch_size)

import os
checkpoint_path = os.path.join(model_dir, "NCF.ckpt")
est.save_checkpoint(checkpoint_path)
# print(model.get_weights())

end = time.time()
print("Time used: ", end - start)

stop_orca_context()
