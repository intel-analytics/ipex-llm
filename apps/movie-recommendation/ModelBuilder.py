import os
import zipfile

import pandas as pd
from bigdl.dllib.feature.dataset import base
from bigdl.orca import OrcaContext
from bigdl.orca import init_orca_context
from bigdl.orca.learn.tf2 import Estimator

OrcaContext.log_output = True

cluster_mode = "local"

if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=1)
elif cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g")

url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
local_file = base.maybe_download('ml-1m.zip', '.', url)
if not os.path.exists('./ml-1m'):
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('.')
    zip_ref.close()

rating_files = "./ml-1m/ratings.dat"
new_rating_files = "./ml-1m/ratings_new.dat"
if not os.path.exists(new_rating_files):
    fin = open(rating_files, "rt")
    fout = open(new_rating_files, "wt")
    for line in fin:
        # replace :: to : for spark 2.4 support
        fout.write(line.replace('::', ':'))
    fin.close()
    fout.close()

spark = OrcaContext.get_spark_session()
df = spark.read.csv(new_rating_files, sep=':', header=True, inferSchema=True).toDF(
    "user", "item", "label", "timestamp")

user_set = df.select('user').collect()
item_set = df.select('item').collect()

min_user_id = min(user_set)[0]
max_user_id = max(user_set)[0]
min_item_id = min(item_set)[0]
max_item_id = max(item_set)[0]
print(min_user_id, max_user_id, min_item_id, max_item_id)

df = df.withColumn('label', df.label - 1)
train_data, test_data = df.randomSplit([0.8, 0.2], 100)


def model_creator(config):
    import tensorflow as tf
    from tensorflow import keras
    embedding_size = 32
    user = keras.layers.Input(dtype=tf.int32, shape=(None,))
    item = keras.layers.Input(dtype=tf.int32, shape=(None,))
    label = keras.layers.Input(dtype=tf.int32, shape=(None,))

    with tf.name_scope("GMF"):
        user_embed_GMF = keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_GMF = keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        GMF = keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_MLP = keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        interaction = tf.concat([user_embed_MLP, item_embed_MLP], axis=-1)
        layer0_MLP = keras.layers.Dense(units=embedding_size * 3, activation='relu')(interaction)
        layer0_MLP = keras.layers.Dropout(rate=0.6)(layer0_MLP)
        layer1_MLP = keras.layers.Dense(units=embedding_size * 2, activation='relu')(layer0_MLP)
        layer1_MLP = keras.layers.Dropout(rate=0.6)(layer1_MLP)
        layer2_MLP = keras.layers.Dense(units=embedding_size, activation='relu')(layer1_MLP)
        layer2_MLP = keras.layers.Dropout(rate=0.6)(layer2_MLP)
        layer3_MLP = keras.layers.Dense(units=embedding_size // 2, activation='relu')(layer2_MLP)
        layer3_MLP = keras.layers.Dropout(rate=0.6)(layer3_MLP)

    # Concate the two parts together
    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = keras.layers.Dense(units=5, activation='softmax')(concatenation)

    model = keras.Model(inputs=[user, item], outputs=outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


batch_size = 1280
epochs = 20
model_dir = './'

# create an Estimator
est = Estimator.from_keras(model_creator=model_creator, workers_per_node=1)

stats = est.fit(train_data,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                steps_per_epoch=800000 // batch_size,
                validation_data=test_data,
                validation_steps=200000 // batch_size)

checkpoint_path = os.path.join(model_dir, "NCF.ckpt")
est.save(checkpoint_path)

# evaluate with Estimator
stats = est.evaluate(test_data,
                     feature_cols=['user', 'item'],
                     label_cols=['label'],
                     num_steps=100000 // batch_size)
# est.shutdown()
print(stats)

pred = est.predict(test_data, feature_cols=['user', 'item'])
df_pred = pred.toPandas()


def f(x):
    x = list(x['prediction'])
    return x.index(max(x))


df_pred['pred_label'] = df_pred.apply(lambda x: f(x), axis=1)
df_pred = df_pred[df_pred["pred_label"] >= 3]
df_pred.rename(columns={'item': 'movie_id'}, inplace=True)
movie_df = pd.read_csv("/content/ml-1m/movies.dat", sep="::", encoding='latin', names=['movie_id', 'name', 'Genre'])
combine_df = df_pred.merge(movie_df, on='movie_id', how='inner')
combine_df = combine_df.groupby('user').agg({'name': pd.Series.to_list, 'Genre':pd.Series.to_list})
combine_df.to_csv("cache.csv")