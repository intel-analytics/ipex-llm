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
from sklearn.model_selection import train_test_split

from bigdl.orca import OrcaContext, init_orca_context, stop_orca_context
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.learn.tf.estimator import Estimator


def build_model(num_users, num_items, layers=[20, 10], include_mf=True, mf_embed=20):
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
        StructField('_c1', IntegerType(), True),
        StructField('_c2', IntegerType(), True)
    ]
)
df = spark.read.csv("{}/{}/ratings.csv".format(data_path, data_type), sep="::",schema=schema,
                     header=False)
df = df.withColumnRenamed('_c0','user').withColumnRenamed('_c1','item').withColumnRenamed('_c2','label')
#print(int(df.describe("item").filter("summary = 'max'").select("item").first().asDict()['item']))
min_user_id = df.agg({"user": "min"}).collect()[0]["min(user)"]
max_user_id = df.agg({"user": "max"}).collect()[0]["max(user)"]
min_item_id = df.agg({"item": "min"}).collect()[0]["min(item)"]
max_item_id = df.agg({"item": "max"}).collect()[0]["max(item)"]
print(min_user_id, max_user_id, min_item_id, max_item_id)
from pyspark.sql import functions
df = df.withColumn('label', functions.lit(1))


import random

data_neg = []
for neg_index in range(min_user_id, max_user_id):
#for neg_index in range(1,2):
    dfs = df.select("item").where("user == %d"%neg_index)
    pdf = set(dfs.toPandas()['item'].values)
    cont = len(dfs.collect()) * 5
    neg_item = random.sample(set(range(min_item_id, max_item_id)) - pdf, cont)
    
    for k in neg_item:
        data_neg.append([neg_index, k, 0])


import pandas as pd

corr_matrix = pd.DataFrame(data_neg)
df_neg = spark.createDataFrame(corr_matrix, schema=schema)
df.unionAll(df_neg)


(train_df, test_df) = df.randomSplit([0.8, 0.2],seed = 11)

model = build_model(max_user_id, max_item_id)
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

estimator = Estimator.from_keras(model)
estimator.fit(train_df,
              batch_size=800,
              epochs=2,
              feature_cols=["user", "item"],
              label_cols=["label"],
              validation_data=test_df)
model = estimator.get_model()
tf.saved_model.save(model, "./model")
# print(model.get_weights())

end = time.time()
print("Time used: ", end - start)

stop_orca_context()
