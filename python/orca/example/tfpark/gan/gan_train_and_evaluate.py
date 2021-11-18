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
from bigdl.dllib.optim.optimizer import MaxIteration
from bigdl.orca.tfpark.gan.gan_estimator import GANEstimator

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.tfpark import TFDataset
from bigdl.orca.tfpark import ZooOptimizer
from bigdl.dllib.utils.common import *
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_gan.examples.mnist.networks import *
from tensorflow_gan.python.losses.losses_impl import *
import tensorflow_datasets as tfds

import os
import argparse

MODEL_DIR = "/tmp/gan_model"
NOISE_DIM = 64

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spatk-submit.')

def eval():

    with tf.Graph().as_default() as g:
        noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(50, NOISE_DIM))
        step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Generator"):
            one_hot = tf.one_hot(tf.concat([tf.range(0, 10)] * 5, axis=0), 10)
            fake_img = conditional_generator((noise, one_hot), is_training=False)
            fake_img = (fake_img * 128.0) + 128.0
            fake_img = tf.cast(fake_img, tf.uint8)
            tiled = tfgan.eval.image_grid(fake_img, grid_shape=(5, 10),
                                          image_shape=(28, 28),
                                          num_channels=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(MODEL_DIR)
            saver.restore(sess, ckpt)
            outputs, step_value = sess.run([tiled, step])
            plt.imsave("./image_{}.png".format(step_value), np.squeeze(outputs), cmap="gray")


if __name__ == "__main__":
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
                "set the environment variable HADOOP_CONF_DIR"
        spark_conf = create_spark_conf().set("spark.executor.memory", "10g") \
            .set("spark.executor.cores", 4) \
            .set("spark.executor.instances", 4) \
            .set("spark.driver.memory", "4g")
        if cluster_mode == "yarn-client":
            sc = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            sc = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    else:
        sc = init_nncontext()

    def input_fn():
        def map_func(data):
            image = data['image']
            label = data['label']
            one_hot_label = tf.one_hot(label, depth=10)
            noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(NOISE_DIM,))
            generator_inputs = (noise, one_hot_label)
            discriminator_inputs = ((tf.to_float(image) / 255.0) - 0.5) * 2
            return (generator_inputs, discriminator_inputs)

        ds = tfds.load("mnist", split="train")
        ds = ds.map(map_func)
        dataset = TFDataset.from_tf_data_dataset(ds, batch_size=36)
        return dataset

    opt = GANEstimator(
        generator_fn=conditional_generator,
        discriminator_fn=conditional_discriminator,
        generator_loss_fn=wasserstein_generator_loss,
        discriminator_loss_fn=wasserstein_discriminator_loss,
        generator_optimizer=ZooOptimizer(tf.train.AdamOptimizer(1e-5, 0.5)),
        discriminator_optimizer=ZooOptimizer(tf.train.AdamOptimizer(1e-4, 0.5)),
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto()
    )

    for i in range(20):
        opt.train(input_fn, MaxIteration(1000))
        eval()

    print("finished...")
    sc.stop()
