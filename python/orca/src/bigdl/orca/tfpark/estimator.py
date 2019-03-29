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
from tensorflow.python.util import function_utils
import tensorflow as tf

from bigdl.optim.optimizer import MaxIteration, Loss, TreeNNAccuracy
from zoo.common import Sample
from zoo.pipeline.api.keras import metrics
from zoo.pipeline.api.net import TFDataset, TFOptimizer, TFNet

from zoo.util import nest
import numpy as np
import six


def add_train_op(model_fn, features, labels, mode, params, config, optimizer):
    model_fn_args = function_utils.fn_args(model_fn)
    kwargs = {}
    if 'labels' in model_fn_args:
        kwargs['labels'] = labels
    else:
        if labels is not None:
            raise ValueError(
                'model_fn does not take labels, but input_fn returns labels.')
    if 'mode' in model_fn_args:
        kwargs['mode'] = mode
    if 'params' in model_fn_args:
        kwargs['params'] = params
    if 'config' in model_fn_args:
        kwargs['config'] = config

    spec = model_fn(features=features, **kwargs)

    if isinstance(spec, tf.estimator.EstimatorSpec):
        train_op = spec.train_op
    else:
        train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN and train_op is None:
        if optimizer is None:
            raise ValueError("optimizer should be set when used for training. For example:" +
                             " Estimator(model_fn, tf.train.AdamOptimizer())")
        grads_and_vars = optimizer.compute_gradients(spec.loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], spec.loss))

        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode,
                                      spec.predictions,
                                      spec.loss,
                                      train_op)


class TFEstimatorSpec:

    def __init__(self, mode, predictions=None, loss=None):
        self.mode = mode
        self.predictions = predictions
        self.loss = loss


class TFEstimator(object):

    def __init__(self, model_fn, optimizer=None, model_dir=None, config=None,
                 params=None, warm_start_from=None):

        def tf_model_fn(features, labels, mode, params, config):
            return add_train_op(model_fn, features, labels, mode, params, config, optimizer)

        estimator = tf.estimator.Estimator(tf_model_fn, model_dir, config, params, warm_start_from)
        self._model_fn = model_fn
        self.optimizer = optimizer
        self.estimator = estimator
        self.config = config
        self.params = params
        self.tf_optimizer = None

    def _call_model_fn(self, features, labels, mode, config):
        model_fn_args = function_utils.fn_args(self._model_fn)
        kwargs = {}
        if 'labels' in model_fn_args:
            kwargs['labels'] = labels
        if 'mode' in model_fn_args:
            kwargs['mode'] = mode
        if 'params' in model_fn_args:
            kwargs['params'] = self.params
        if 'config' in model_fn_args:
            kwargs['config'] = config

        model_fn_results = self._model_fn(features=features, **kwargs)

        if not isinstance(model_fn_results, TFEstimatorSpec):
            raise ValueError('model_fn should return an TFEstimatorSpec.')

        return model_fn_results

    def train(self, input_fn, steps=None):

        with tf.Graph().as_default() as g:
            global_step_tensor = self.estimator._create_and_assert_global_step(g)
            add_step_input = tf.placeholder(dtype=tf.int64, shape=())
            assign_step = tf.assign_add(global_step_tensor, add_step_input)
            result = self.estimator._call_input_fn(input_fn, tf.estimator.ModeKeys.TRAIN)
            if isinstance(result, TFDataset):
                if not result.has_batch:
                    raise ValueError("The batch_size of TFDataset must be " +
                                     "specified when used for training.")
                spec = self._call_model_fn(result.feature_tensors,
                                           result.label_tensors,
                                           tf.estimator.ModeKeys.TRAIN,
                                           self.config)
                optim_method = TFOptimizer.to_bigdl_optim_method(koptim_method=self.optimizer)
                latest_checkpoint = self.estimator.latest_checkpoint()

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    if latest_checkpoint:
                        saver.restore(sess, latest_checkpoint)
                    else:
                        sess.run(tf.global_variables_initializer())

                    opt = TFOptimizer.from_loss(spec.loss, optim_method, session=sess)
                    opt.optimize(MaxIteration(steps))
                    sess.run(assign_step, feed_dict={add_step_input: steps})
                    final_step = sess.run(global_step_tensor)
                    saver.save(sess, self.estimator.model_dir + "/model", global_step=final_step)
                    return self

        return self.estimator.train(input_fn, steps=steps)

    def evaluate(self, input_fn, eval_methods, steps=None, checkpoint_path=None):
        if not all(isinstance(metric, six.string_types) for metric in eval_methods):
            raise ValueError("All metrics should be string types")
        with tf.Graph().as_default() as g:
            result = self.estimator._call_input_fn(input_fn, tf.estimator.ModeKeys.EVAL)
            if isinstance(result, TFDataset):
                spec = self._call_model_fn(result.feature_tensors,
                                           result.label_tensors,
                                           tf.estimator.ModeKeys.PREDICT,
                                           self.config)
                latest_checkpoint = self.estimator.latest_checkpoint()

                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint

                with tf.Session() as sess:
                    if checkpoint_path:
                        saver = tf.train.Saver()
                        saver.restore(sess, checkpoint_path)
                    else:
                        sess.run(tf.global_variables_initializer())
                    inputs = nest.flatten(result._original_tensors[0])
                    outputs = nest.flatten(spec.predictions)
                    tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

                    rdd = result.rdd.map(lambda t: Sample.from_ndarray(nest.flatten(t[0]),
                                                                       nest.flatten(t[1])))
                    if result.batch_per_thread < 0:
                        batch_size = result.batch_size
                    else:
                        batch_size = result.batch_per_thread * result.rdd.getNumPartitions()

                    eval_methods = [self._to_bigdl_metric(m) for m in eval_methods]
                    results = tfnet.evaluate(rdd, batch_size, eval_methods)
                    final_result = dict([(r.method, r.result) for r in results])
                    return final_result

        return self.estimator.evaluate(input_fn, steps, checkpoint_path=checkpoint_path)

    def predict(self, input_fn, checkpoint_path=None):
        with tf.Graph().as_default() as g:
            result = self.estimator._call_input_fn(input_fn, tf.estimator.ModeKeys.PREDICT)
            if isinstance(result, TFDataset):
                spec = self._call_model_fn(result.feature_tensors,
                                           None,
                                           tf.estimator.ModeKeys.PREDICT,
                                           self.config)
                latest_checkpoint = self.estimator.latest_checkpoint()

                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint

                with tf.Session() as sess:
                    if checkpoint_path:
                        saver = tf.train.Saver()
                        saver.restore(sess, checkpoint_path)
                    else:
                        sess.run(tf.global_variables_initializer())
                    inputs = nest.flatten(result._original_tensors[0])
                    outputs = nest.flatten(spec.predictions)
                    tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

                    rdd = result.rdd.map(lambda t:
                                         Sample.from_ndarray(nest.flatten(t), np.array([0.0])))

                    results = tfnet.predict(rdd, result.batch_per_thread)
                    return results

        return self.estimator.predict(input_fn, checkpoint_path=checkpoint_path)

    @staticmethod
    def _to_bigdl_metric(metric):
        metric = metric.lower()
        if metric == "accuracy" or metric == "acc":
            return metrics.Accuracy()
        elif metric == "top5accuracy" or metric == "top5acc":
            return metrics.Top5Accuracy()
        elif metric == "mae":
            from bigdl.optim.optimizer import MAE
            return MAE()
        elif metric == "auc":
            return metrics.AUC()
        elif metric == "loss":
            return Loss()
        elif metric == "treennaccuracy":
            return TreeNNAccuracy()
        else:
            raise TypeError("Unsupported metric: %s" % metric)
