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

import math
import argparse

import tensorflow as tf

from pyspark.sql.types import StructField, StructType, IntegerType, ArrayType
from pyspark.sql.functions import udf, lit, collect_list, explode

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator

parser = argparse.ArgumentParser(description='TensorFlow NCF Training')
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--memory", type=str, required=True)
args = parser.parse_args()


def model_creator(config):
    embedding_size = config["embedding_size"]
    user = tf.keras.layers.Input(dtype=tf.int32, shape=(None,))
    item = tf.keras.layers.Input(dtype=tf.int32, shape=(None,))

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

    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = tf.keras.Model(inputs=[user, item], outputs=outputs)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    return model


sc = init_orca_context(cluster_mode="local", cores="*")
spark = OrcaContext.get_spark_session()

schema = StructType([StructField('user', IntegerType(), True),
                     StructField('item', IntegerType(), True)])
# Need spark3 to support delimiter with more than one character.
df = spark.read.csv("./ml-1m/ratings.dat", sep="::", schema=schema, header=False)
df = df.withColumn('label', lit(1))

min_user_id = df.agg({"user": "min"}).collect()[0]["min(user)"]
max_user_id = df.agg({"user": "max"}).collect()[0]["max(user)"]
min_item_id = df.agg({"item": "min"}).collect()[0]["min(item)"]
max_item_id = df.agg({"item": "max"}).collect()[0]["max(item)"]
print(min_user_id, max_user_id, min_item_id, max_item_id)


def neg_sample(x):
    import random
    neg_scale = 4
    item_count = len(x) * neg_scale
    max_count = max_item_id - len(set(x))
    neg_count = min(item_count, max_count)
    neg_item = random.sample(set(range(min_item_id, max_item_id+1)) - set(x), neg_count)
    return neg_item

neg_sample_udf = udf(neg_sample, ArrayType(IntegerType(), False))

df_neg = df.groupBy('user').agg(neg_sample_udf(collect_list('item')).alias('item_list'))
df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
df_neg = df_neg.withColumn('label', lit(0))
df_neg = df_neg.withColumnRenamed('col', 'item')
df = df.unionAll(df_neg)

train_df, test_df = df.randomSplit([0.8, 0.2], 100)

batch_size = 256
epochs = 5

steps_per_epoch = math.ceil(train_df.count() / batch_size)
val_steps = math.ceil(test_df.count() / batch_size)

# create an Estimator
backend = 'spark'
est = Estimator.from_keras(model_creator=model_creator,
                           config={"embedding_size": 16},
                           backend=backend)

stats = est.fit(train_df,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                steps_per_epoch=steps_per_epoch,
                validation_data=test_df,
                validation_steps=val_steps)

# save model in H5 format
est.save("./ncf.h5")

# evaluate with Estimator
stats = est.evaluate(test_df,
                     feature_cols=['user', 'item'],
                     label_cols=['label'],
                     batch_size=batch_size,
                     num_steps=val_steps)
print("Evaluation results:")
print(stats)
est.shutdown()

stop_orca_context()
