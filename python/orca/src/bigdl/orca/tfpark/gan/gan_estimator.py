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
import inspect
import tempfile
import os
import tensorflow as tf
import numpy as np

from zoo.pipeline.api.net.tf_optimizer import TFModel
from zoo.tfpark import TFOptimizer
from zoo.tfpark.gan.common import GanOptimMethod

# todo make it inherit Estimator


class GANEstimator(object):

    def __init__(self,
                 generator_fn,
                 discriminator_fn,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 generator_optimizer,
                 discriminator_optimizer,
                 generator_steps=1,
                 discriminator_steps=1,
                 model_dir=None,
                 ):
        self._generator_fn = generator_fn
        self._discriminator_fn = discriminator_fn
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_steps = generator_steps
        self._discriminator_steps = discriminator_steps
        self._generator_optim_method = generator_optimizer
        self._discriminator_optim_method = discriminator_optimizer

        if model_dir is None:
            folder = tempfile.mkdtemp()
            self.checkpoint_path = os.path.join(folder, "gan_model")
        else:
            self.checkpoint_path = model_dir

    @staticmethod
    def _call_fn_maybe_with_counter(fn, counter, *args):
        fn_args = inspect.getargspec(fn).args
        if "counter" in fn_args:
            return fn(*args, counter=counter)
        else:
            return fn(*args)

    def train(self, dataset, end_trigger):

        with tf.Graph().as_default() as g:

            generator_inputs = dataset.tensors[0]
            real_data = dataset.tensors[1]

            counter = tf.Variable(0, dtype=tf.int32)

            period = self._discriminator_steps + self._generator_steps

            is_discriminator_phase = tf.less(tf.mod(counter, period), self._discriminator_steps)

            with tf.variable_scope("generator"):
                gen_data = self._call_fn_maybe_with_counter(self._generator_fn, counter,
                                                            generator_inputs)

            with tf.variable_scope("discriminator"):
                fake_d_outputs = self._call_fn_maybe_with_counter(self._discriminator_fn,
                                                                  counter,
                                                                  gen_data, generator_inputs)

            with tf.variable_scope("discriminator", reuse=True):
                real_d_outputs = self._call_fn_maybe_with_counter(self._discriminator_fn,
                                                                  counter,
                                                                  real_data, generator_inputs)

            with tf.name_scope("generator_loss"):
                generator_loss = self._call_fn_maybe_with_counter(self._generator_loss_fn,
                                                                  counter,
                                                                  fake_d_outputs)

            with tf.name_scope("discriminator_loss"):
                discriminator_loss = self._call_fn_maybe_with_counter(self._discriminator_loss_fn,
                                                                      counter,
                                                                      real_d_outputs,
                                                                      fake_d_outputs)

            generator_variables = tf.trainable_variables("generator")
            generator_grads = tf.gradients(generator_loss, generator_variables)
            discriminator_variables = tf.trainable_variables("discriminator")
            discriminator_grads = tf.gradients(discriminator_loss, discriminator_variables)

            variables = generator_variables + discriminator_variables

            def true_fn():
                return [tf.zeros_like(grad) for grad in generator_grads]

            def false_fn():
                return generator_grads

            g_grads = tf.cond(is_discriminator_phase, true_fn=true_fn, false_fn=false_fn)
            d_grads = tf.cond(is_discriminator_phase, lambda: discriminator_grads,
                              lambda: [tf.zeros_like(grad) for grad in discriminator_grads])
            loss = tf.cond(is_discriminator_phase,
                           lambda: discriminator_loss,
                           lambda: generator_loss)

            grads = g_grads + d_grads

            with tf.control_dependencies(grads):
                increase_counter = tf.assign_add(counter, 1)

            g_param_size = sum([np.product(g.shape) for g in g_grads])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf_model = TFModel.create_for_unfreeze(loss, sess,
                                                       inputs=dataset._original_tensors,
                                                       grads=grads,
                                                       variables=variables,
                                                       graph=g,
                                                       tensors_with_value=None,
                                                       session_config=None,
                                                       metrics=None,
                                                       updates=[increase_counter],
                                                       model_dir=self.checkpoint_path)

                optimizer = TFOptimizer(tf_model, GanOptimMethod(self._discriminator_optim_method,
                                                                 self._generator_optim_method,
                                                                 g_param_size.value,
                                                                 self._discriminator_steps,
                                                                 self._generator_steps),
                                        sess=sess,
                                        dataset=dataset, model_dir=self.checkpoint_path)
                optimizer.optimize(end_trigger)
                steps = sess.run(counter)
                saver = tf.train.Saver()
                saver.save(optimizer.sess, self.checkpoint_path, global_step=steps)
