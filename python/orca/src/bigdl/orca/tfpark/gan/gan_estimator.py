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
import os
import tempfile

import tensorflow as tf

from zoo.tfpark import TFOptimizer
# todo make it inherit Estimator
from zoo.tfpark.zoo_optimizer import FakeOptimMethod
from zoo.util import nest


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
                 session_config=None,
                 ):
        from zoo.tfpark import ZooOptimizer
        assert isinstance(generator_optimizer, ZooOptimizer),\
            "generator_optimizer should be a ZooOptimizer"
        assert isinstance(discriminator_optimizer, ZooOptimizer),\
            "discriminator_optimizer should be a ZooOptimizer"
        self._generator_fn = generator_fn
        self._discriminator_fn = discriminator_fn
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_steps = generator_steps
        self._discriminator_steps = discriminator_steps
        self._gen_opt = generator_optimizer
        self._dis_opt = discriminator_optimizer
        self._session_config = session_config

        if model_dir is None:
            folder = tempfile.mkdtemp()
            self.checkpoint_path = os.path.join(folder, "model")
            self.model_dir = folder
        else:
            self.checkpoint_path = os.path.join(model_dir, "model")
            self.model_dir = model_dir

    @staticmethod
    def _call_fn_maybe_with_counter(fn, counter, *args):
        fn_args = inspect.getargspec(fn).args
        if "counter" in fn_args:
            return fn(*args, counter=counter)
        else:
            return fn(*args)

    def train(self, input_fn, end_trigger):

        with tf.Graph().as_default() as g:

            dataset = input_fn()

            generator_inputs = dataset.tensors[0]
            real_data = dataset.tensors[1]

            counter = tf.train.get_or_create_global_step()

            period = self._discriminator_steps + self._generator_steps

            is_discriminator_phase = tf.less(tf.mod(counter, period), self._discriminator_steps)

            with tf.variable_scope("Generator"):
                gen_data = self._call_fn_maybe_with_counter(self._generator_fn, counter,
                                                            generator_inputs)

            with tf.variable_scope("Discriminator"):
                fake_d_outputs = self._call_fn_maybe_with_counter(self._discriminator_fn,
                                                                  counter,
                                                                  gen_data, generator_inputs)

            with tf.variable_scope("Discriminator", reuse=True):
                real_d_outputs = self._call_fn_maybe_with_counter(self._discriminator_fn,
                                                                  counter,
                                                                  real_data, generator_inputs)

            with tf.name_scope("Generator_loss"):
                generator_loss = self._call_fn_maybe_with_counter(self._generator_loss_fn,
                                                                  counter,
                                                                  fake_d_outputs)
                gen_reg_loss = tf.losses.get_regularization_loss("Generator")

                generator_loss = generator_loss + gen_reg_loss

            with tf.name_scope("Discriminator_loss"):
                discriminator_loss = self._call_fn_maybe_with_counter(self._discriminator_loss_fn,
                                                                      counter,
                                                                      real_d_outputs,
                                                                      fake_d_outputs)
                dis_reg_loss = tf.losses.get_regularization_loss("Discriminator")
                discriminator_loss = discriminator_loss + dis_reg_loss

            generator_variables = tf.trainable_variables("Generator")
            discriminator_variables = tf.trainable_variables("Discriminator")

            def run_gen_compute():
                gen_grads_vars = self._gen_opt.compute_gradients(generator_loss,
                                                                 var_list=generator_variables)
                gen_grads = [grad for grad, var in gen_grads_vars]
                dis_grads = [tf.zeros_like(var) for var in discriminator_variables]

                return gen_grads + dis_grads

            def run_dis_compute():
                dis_grads_vars = self._gen_opt.compute_gradients(discriminator_loss,
                                                                 var_list=discriminator_variables)
                dis_grads = [grad for grad, var in dis_grads_vars]
                gen_gards = [tf.zeros_like(var) for var in generator_variables]
                return gen_gards + dis_grads

            grads = tf.cond(is_discriminator_phase, run_dis_compute, run_gen_compute)

            grads_vars = list(zip(grads, generator_variables + discriminator_variables))

            gen_grads_vars = grads_vars[:len(generator_variables)]
            dis_grads_vars = grads_vars[len(generator_variables):]

            grads = [grad for grad, var in grads_vars]

            _train_op = tf.cond(is_discriminator_phase,
                                lambda: self._dis_opt.apply_gradients(dis_grads_vars),
                                lambda: self._gen_opt.apply_gradients(gen_grads_vars))

            variables = generator_variables + discriminator_variables

            loss = tf.cond(is_discriminator_phase,
                           lambda: discriminator_loss,
                           lambda: generator_loss)

            with tf.control_dependencies([_train_op]):
                increase_counter = tf.assign_add(counter, 1)

            with tf.control_dependencies([increase_counter]):
                train_op = tf.no_op()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                kpt = tf.train.latest_checkpoint(self.model_dir)
                if kpt is not None:
                    saver.restore(sess, kpt)
                opt = TFOptimizer._from_grads(loss, sess,
                                              inputs=nest.flatten(dataset._original_tensors),
                                              labels=[],
                                              grads=grads, variables=variables, dataset=dataset,
                                              optim_method=FakeOptimMethod(),
                                              session_config=self._session_config,
                                              model_dir=os.path.join(self.model_dir, "tmp"),
                                              train_op=train_op)
                opt.optimize(end_trigger)
                saver = tf.train.Saver()
                saver.save(sess, self.checkpoint_path, global_step=counter)
