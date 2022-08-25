import os
import zipfile
import argparse

import numpy as np
import tensorflow as tf

from bigdl.dllib.feature.dataset import base
from sklearn.model_selection import train_test_split

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.orca.data import SharedValue
import bigdl.orca.data.pandas
import argparse

# define arg parser object
parser = argparse.ArgumentParser(description='ncf dataframe programming')
parser.add_argument('--cluster_mode', type=str, default='local', help='Optional values: local, yarn, k8s.')
parser.add_argument('--master', type=str, default='master', help='In k8s mode, the parameter master must be passed in.')
parser.add_argument('--image_name_k8s', type=str, default='image_name_k8s', help='In k8s mode, the parameter image_name_k8s must be passed in.')

args = parser.parse_args()
cluster_mode = args.cluster_mode
master = args.master
image_name_k8s = args.image_name_k8s

# recommended to set it to True when running Analytics Zoo in Jupyter notebook
OrcaContext.log_output = True # (this will display terminal's stdout and stderr in the Jupyter notebook).

if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=1) # run in local mode
elif cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g") # run on Hadoop YARN cluster
elif cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", master=master,
            container_image=image_name_k8s, num_nodes=1, memory="128g", cores=4) # run in local mode

print("INFO 1 cluster_mode_init_success!")

# Read in the dataset, and do a little preprocessing
new_rating_files="/ppml/trusted-big-data-ml/work/data/ml-1m/ratings_new.dat.2"
if not os.path.exists(new_rating_files):
        print("INFO ERROR ratings_new.dat does not exist")
        exit(1)

# read csv
spark = OrcaContext.get_spark_session()
df = spark.read.csv(new_rating_files, sep=':', header=True, inferSchema=True).toDF(
"user", "item", "label", "timestamp")
user_set = df.select('user').collect()
item_set = df.select('item').collect()
#min_user_id = min(user_set)[0]
max_user_id = max(user_set)[0]
#min_item_id = min(item_set)[0]
max_item_id = max(item_set)[0]
#print(min_user_id, max_user_id, min_item_id, max_item_id)
# update label starting from 0
df = df.withColumn('label', df.label-1)
# split to train/test dataset
train_data, test_data = df.randomSplit([0.8, 0.2], 100)


class NCF(object):
    def __init__(self, embed_size, user_size, item_size):
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))
        # GMF part starts
        with tf.name_scope("GMF"):
            user_embed_GMF = tf.contrib.layers.embed_sequence(self.user, vocab_size=user_size + 1,
                                                              embed_dim=embed_size)
            item_embed_GMF = tf.contrib.layers.embed_sequence(self.item, vocab_size=item_size + 1,
                                                              embed_dim=embed_size)
            GMF = tf.multiply(user_embed_GMF, item_embed_GMF)
        # MLP part starts
        with tf.name_scope("MLP"):
            user_embed_MLP = tf.contrib.layers.embed_sequence(self.user, vocab_size=user_size + 1,
                                                              embed_dim=embed_size)
            item_embed_MLP = tf.contrib.layers.embed_sequence(self.item, vocab_size=item_size + 1,
                                                              embed_dim=embed_size)
            interaction = tf.concat([user_embed_MLP, item_embed_MLP], axis=-1)
            layer1_MLP = tf.layers.dense(inputs=interaction, units=embed_size * 2)
            layer1_MLP = tf.layers.dropout(layer1_MLP, rate=0.2)
            layer2_MLP = tf.layers.dense(inputs=layer1_MLP, units=embed_size)
            layer2_MLP = tf.layers.dropout(layer2_MLP, rate=0.2)
            layer3_MLP = tf.layers.dense(inputs=layer2_MLP, units=embed_size // 2)
            layer3_MLP = tf.layers.dropout(layer3_MLP, rate=0.2)
        # Concate the two parts together
        with tf.name_scope("concatenation"):
            concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
            self.logits = tf.layers.dense(inputs=concatenation, units=5)
            self.logits_softmax = tf.nn.softmax(self.logits)
            self.class_number = tf.argmax(self.logits_softmax, 1)
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits, name='loss'))
        with tf.name_scope("optimzation"):
            self.optim = tf.train.AdamOptimizer(1e-3, name='Adam')
            self.optimizer = self.optim.minimize(self.loss)
embedding_size=16
model = NCF(embedding_size, max_user_id, max_item_id)
print("INFO NCF model defined success!")

batch_size=1280
epochs=1
model_dir='./logs-ncf'

# create an Estimator.
estimator = Estimator.from_graph(
            inputs=[model.user, model.item],
            outputs=[model.class_number],
            labels=[model.label],
            loss=model.loss,
            optimizer=model.optim,
            model_dir=model_dir,
            metrics={"loss": model.loss})
print("INFO estimator created success!")

estimator.fit(data=train_data,
              batch_size=batch_size,
              epochs=epochs,
              feature_cols=['user', 'item'],
              label_cols=['label'],
              validation_data=test_data)
print("INFO estimator fit success!")

checkpoint_path = os.path.join(model_dir, "NCF.ckpt")
estimator.save_tf_checkpoint(checkpoint_path)
estimator.shutdown()
print("INFO estimator.save checkpoint success!")

stop_orca_context()
