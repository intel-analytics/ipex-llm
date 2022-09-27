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

import sys
import time
import os

import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator


def model_creator(config):
    embedding_size=16
    user = tf.keras.layers.Input(dtype=tf.int32, shape=(None,))
    item = tf.keras.layers.Input(dtype=tf.int32, shape=(None,))
    label = tf.keras.layers.Input(dtype=tf.int32, shape=(None,))

    with tf.name_scope("GMF"):
        user_embed_GMF = tf.keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_GMF = tf.keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        GMF = tf.keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = tf.keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_MLP = tf.keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        interaction = tf.concat([user_embed_MLP, item_embed_MLP], axis=-1)
        layer1_MLP = tf.keras.layers.Dense(units=embedding_size * 2, activation='relu')(interaction)
        layer1_MLP = tf.keras.layers.Dropout(rate=0.2)(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dense(units=embedding_size, activation='relu')(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dropout(rate=0.2)(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dense(units=embedding_size // 2, activation='relu')(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dropout(rate=0.2)(layer3_MLP)

    # Concate the two parts together
    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = tf.keras.Model(inputs=[user, item], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

cluster_mode = str(sys.argv[1])
# cluster_mode = "k8s"
if cluster_mode == "local":
    sc = init_orca_context(memory=str(sys.argv[2]))
elif cluster_mode == "standalone":
    sc = init_orca_context("standalone", cores=int(sys.argv[3]), num_nodes=int(sys.argv[4]), memory=str(sys.argv[2]))
elif cluster_mode == "yarn":
    sc = init_orca_context("yarn", cores=int(sys.argv[3]), num_nodes=int(sys.argv[4]), memory=str(sys.argv[2]))
elif cluster_mode == "k8s":
    sc = init_orca_context(cluster_mode="k8s", cores=int(sys.argv[3]), num_nodes=int(sys.argv[4]), memory=str(sys.argv[2]),
                           master="k8s://https://172.16.0.200:6443",
                           container_image="10.239.45.10/arda/intelanalytics/bigdl-k8s-spark-3.1.2:0.14.0-SNAPSHOT",
                           conf={"spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data"})
elif cluster_mode == "spark-submit":  # To test k8s using spark-submit
    sc = init_orca_context(cluster_mode="spark-submit")


data_path = "./bigdl2.0/data"
data_type = "ml-1m"
# Need spark3 to support delimiter with more than one character.
spark= OrcaContext.get_spark_session()
from pyspark.sql.types import StructField, StructType, IntegerType

schema = StructType(
    [
        StructField('user', IntegerType(), True),
        StructField('item', IntegerType(), True)
    ]
)
df = spark.read.csv("{}/{}/ratings.csv".format(data_path, data_type), sep="::",schema=schema,
                     header=False)
min_user_id = df.agg({"user": "min"}).collect()[0]["min(user)"]
max_user_id = df.agg({"user": "max"}).collect()[0]["max(user)"]
min_item_id = df.agg({"item": "min"}).collect()[0]["min(item)"]
max_item_id = df.agg({"item": "max"}).collect()[0]["max(item)"]
print(min_user_id, max_user_id, min_item_id, max_item_id)
from pyspark.sql import functions
df = df.withColumn('label', functions.lit(1))

from pyspark.sql import functions as F
from pyspark.sql import types as T
import random

def neg_sample(x):
    item_count = len(x) * 4
    max_count = max_item_id - len(set(x))
    neg_item = random.sample(set(range(min_item_id, max_item_id+1)) - set(x), min(item_count,max_count))
    return neg_item

neg_sample_udf = F.udf(neg_sample, T.ArrayType(IntegerType(), False))

df_neg= df.groupBy('user').agg(neg_sample_udf(F.collect_list('item')).alias('item_list'))
from pyspark.sql.functions import *
df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
df_neg = df_neg.withColumn('label', functions.lit(0))
df = df.unionAll(df_neg)
num_sample = df.count()
train_df, test_df = df.randomSplit([0.8, 0.2],100)


batch_size=1280
epochs=5
model_dir='./'

# create an Estimator
# backend = 'spark'
# est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1, backend=backend, model_dir=model_dir)
est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1)

stats = est.fit(train_df,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                steps_per_epoch=int(train_df.count() // batch_size),
                validation_data=train_df,
                validation_steps =int(train_df.count() // batch_size))

checkpoint_path = os.path.join(model_dir, "NCF.ckpt")
est.save_checkpoint(checkpoint_path)


# evaluate with Estimator
stats = est.evaluate(train_df, 
                     feature_cols=['user', 'item'],
                     label_cols=['label'],
                     batch_size=batch_size,
                     num_steps=int(epochs*train_df.count() // batch_size))

print(stats)
est.shutdown()

stop_orca_context()
