#
# Copyright 2018 Analytics Zoo Authors.
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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_gan.examples.mnist.networks import *

noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(20, 10))

with tf.variable_scope("generator"):
    fake_img = unconditional_generator(noise=noise)
    tiled = tfgan.eval.image_grid(fake_img, grid_shape=(2, 10),
                                  image_shape=(28, 28),
                                  num_channels=1)

with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/gan_model/model-5000")
    outputs = sess.run(tiled)

    plt.axis('off')
    plt.imshow(np.squeeze(outputs))
    plt.show()
