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
import argparse

import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator

parser = argparse.ArgumentParser(description='Tensorflow ImageNet Training')
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--memory", type=str, required=True)
args = parser.parse_args()


def model_creator(config):
    embedding_size = 16
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
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    return model

cluster_mode = args.mode
if cluster_mode == "local":
    sc = init_orca_context(memory=args.memory)


data_path = "."
data_type = "ml-1m"
# Need spark3 to support delimiter with more than one character.
spark = OrcaContext.get_spark_session()
from pyspark.sql.types import StructField, StructType, IntegerType, StringType

schema = StructType(
    [
        StructField('user', IntegerType(), True),
        StructField('item', IntegerType(), True)
    ]
)
schema_user = StructType(
    [
        StructField('user_u', IntegerType(), True),
        StructField('gender', StringType(), True),
        StructField('age', IntegerType(), True),
        StructField('occupation', IntegerType(), True),
        StructField('zipcode', IntegerType(), True)
    ]
)
schema_item = StructType(
    [
        StructField('item_i', IntegerType(), True),
        StructField('title', StringType(), True),
        StructField('genres', StringType(), True)
    ]
)
df = spark.read.csv("{}/{}/ratings.dat".format(data_path, data_type), sep="::", schema=schema,
                    header=False)
min_user_id = df.agg({"user": "min"}).collect()[0]["min(user)"]
max_user_id = df.agg({"user": "max"}).collect()[0]["max(user)"]
min_item_id = df.agg({"item": "min"}).collect()[0]["min(item)"]
max_item_id = df.agg({"item": "max"}).collect()[0]["max(item)"]
print(min_user_id, max_user_id, min_item_id, max_item_id)
from pyspark.sql import functions
df = df.withColumn('label', functions.lit(1))

df_user = spark.read.csv("{}/{}/users.dat".format(data_path, data_type), sep="::", schema=schema_user,
                     header=False)
df_item = spark.read.csv("{}/{}/movies.dat".format(data_path, data_type), sep="::", schema=schema_item,
                     header=False)
from pyspark.ml.feature import StringIndexer
indexer_u = StringIndexer(inputCol="gender", outputCol="genderindex").fit(df_user)
df_user = indexer_u.transform(df_user)
df_user = df_user.withColumn("genderindex", df_user["genderindex"].cast(IntegerType()))
indexer_i = StringIndexer(inputCol="title", outputCol="titleindex").fit(df_item)
df_item = indexer_i.transform(df_item)
indexer_i = StringIndexer(inputCol="genres", outputCol="genresindex").fit(df_item)
df_item = indexer_i.transform(df_item)
df_item = df_item.withColumn("titleindex", df_item["titleindex"].cast(IntegerType())).withColumn("genresindex", df_item["genresindex"].cast(IntegerType()))
df_user = df_user.drop('gender')
df_item = df_item.drop('title', 'genres')
df_feat = df.join(df_user, df.user == df_user.user_u, "inner")
df_feat = df_feat.join(df_item, df_feat.item == df_item.item_i, "inner")
df_feat = df_feat.drop('user_u', 'item_i')

from pyspark.sql.functions import udf, collect_list
from pyspark.sql.types import ArrayType
import random

neg_scale = 4


def neg_sample(x):
    item_count = len(x) * neg_scale
    max_count = max_item_id - len(set(x))
    neg_count = min(item_count, max_count)
    neg_item = random.sample(set(range(min_item_id, max_item_id+1)) - set(x), neg_count)
    return neg_item

neg_sample_udf = udf(neg_sample, ArrayType(IntegerType(), False))

df_neg = df.groupBy('user').agg(neg_sample_udf(collect_list('item')).alias('item_list'))
from pyspark.sql.functions import *
df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
df_neg = df_neg.withColumn('label', functions.lit(0))
df_neg = df_neg.withColumnRenamed('col', 'item')
df = df.unionAll(df_neg)
num_sample = df.count()
train_df, test_df = df.randomSplit([0.8, 0.2], 100)


batch_size = 256
epochs = 5

# create an Estimator
backend = 'spark'
# est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1, backend=backend)
est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1)

stats = est.fit(train_df,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                steps_per_epoch=int(train_df.count() // batch_size),
                validation_data=test_df,
                validation_steps=int(test_df.count() // batch_size))

# save model in H5 format
est.save("./ncf_tf_model.h5")

# evaluate with Estimator
stats = est.evaluate(test_df,
                     feature_cols=['user', 'item'],
                     label_cols=['label'],
                     batch_size=batch_size,
                     num_steps=int(test_df.count()*epochs // batch_size))

print(stats)
est.shutdown()

stop_orca_context()
