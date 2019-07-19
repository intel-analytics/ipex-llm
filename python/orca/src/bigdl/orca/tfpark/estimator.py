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
        """
        :param model_fn: Model function. Follows the signature:

            * Args:

                * `features`: This is the first item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same.
                * `labels`: This is the second item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same (for multi-head models).
                    If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                    be passed. If the `model_fn`'s signature does not accept
                    `mode`, the `model_fn` must still be able to handle
                    `labels=None`.
                * `mode`: Optional. Specifies if this training, evaluation or
                    prediction. See `tf.estimator.ModeKeys`.
                * `params`: Optional `dict` of hyperparameters.  Will receive what
                    is passed to Estimator in `params` parameter. This allows
                    to configure Estimators from hyper parameter tuning.
                * `config`: Optional `estimator.RunConfig` object. Will receive what
                    is passed to Estimator as its `config` parameter, or a default
                    value. Allows setting up things in your `model_fn` based on
                    configuration such as `num_ps_replicas`, or `model_dir`.

            * Returns:
                `zoo.tfpark.estimator.TFEstimatorSpec`
        :param optimizer: the tf.train.Optimizer to be used in training,
                         e.g. tf.train.AdamOptimizer()
        :param model_dir: Directory to save model parameters, graph and etc. This can
            also be used to load checkpoints from the directory into an estimator to
            continue training a previously saved model. If `PathLike` object, the
            path will be resolved. If `None`, the model_dir in `config` will be used
            if set. If both are set, they must be same. If both are `None`, a
            temporary directory will be used.
        :param config: `estimator.RunConfig` configuration object.
        :param params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
        :param warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                       warm-start from, or a `tf.estimator.WarmStartSettings`
                       object to fully configure warm-starting.  If the string
                       filepath is provided instead of a
                       `tf.estimator.WarmStartSettings`, then all variables are
                       warm-started, and it is assumed that vocabularies
                       and `tf.Tensor` names are unchanged.
        """

        def tf_model_fn(features, labels, mode, params, config):
            return add_train_op(model_fn, features, labels, mode, params, config, optimizer)

        estimator = tf.estimator.Estimator(tf_model_fn, model_dir, config, params, warm_start_from)
        self._model_fn = model_fn
        self.optimizer = optimizer
        self.estimator = estimator
        self.config = config
        self.params = params
        self.tf_optimizer = None
        self.gradient_clipping_norm = None
        self.gradient_clipping_constant = None
        self.train_summary = None
        self.val_summary = None

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

    def clear_gradient_clipping(self):
        self.gradient_clipping_constant = None
        self.gradient_clipping_norm = None

    def set_constant_gradient_clipping(self, min, max):
        """
        Configure constant clipping settings.


        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        self.gradient_clipping_constant = (min, max)

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        """
        Configure L2 norm clipping settings.


        :param clip_norm: gradient L2-Norm threshold
        """
        self.gradient_clipping_norm = clip_norm

    def set_train_summary(self, summary):
        """
        Set training summary for visualization.

        :param summary: bigdl.optim.optimizer.TrainSummary
        """
        self.train_summary = summary

    def set_val_summary(self, summary):
        """
        Set validation summary for visualization.

        :param summary: bigdl.optim.optimizer.ValidationSummary
        """
        self.val_summary = summary

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, input_fn, steps=None):
        """Trains a model given training data `input_fn`.

        :param input_fn: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
        :param steps: Number of steps for which to train the model.

        Returns:
          `self`, for chaining.
        """

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

                    opt = TFOptimizer.from_loss(spec.loss,
                                                optim_method,
                                                session=sess,
                                                clip_norm=self.gradient_clipping_norm,
                                                clip_value=self.gradient_clipping_constant)

                    if self.train_summary is not None:
                        opt.set_train_summary(self.train_summary)

                    if self.val_summary is not None:
                        opt.set_val_summary(self.val_summary)

                    opt.optimize(MaxIteration(steps))
                    sess.run(assign_step, feed_dict={add_step_input: steps})
                    final_step = sess.run(global_step_tensor)
                    saver.save(sess, self.estimator.model_dir + "/model", global_step=final_step)
                    return self

        return self.estimator.train(input_fn, steps=steps)

    def evaluate(self, input_fn, eval_methods, steps=None, checkpoint_path=None):
        """Evaluates the model given evaluation data `input_fn`.

        :param input_fn: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
        :param eval_methods: a list of strings to specify the evaluation metrics to
                            be used in this model
        :param steps: Number of steps for which to evaluate model.
        :param checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, evaluation is run with newly initialized `Variables`
            instead of ones restored from checkpoint.

        Returns:
          A dict containing the evaluation metrics specified in `model_fn` keyed by
          name.
        """
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

                    data = result.get_evaluation_data()
                    if result.batch_per_thread < 0:
                        batch_size = result.batch_size
                    else:
                        batch_size = result.batch_per_thread * result.get_num_partitions()

                    eval_methods = [self._to_bigdl_metric(m) for m in eval_methods]
                    results = tfnet.evaluate(data, batch_size, eval_methods)
                    final_result = dict([(r.method, r.result) for r in results])
                    return final_result

        return self.estimator.evaluate(input_fn, steps, checkpoint_path=checkpoint_path)

    def predict(self, input_fn, checkpoint_path=None):
        """Outputs predictions for given features.

        :param input_fn: A function that constructs the features.
              * A `TFDataset` object, each elements of which is a tuple `(features, None)`.
              * A `tf.data.Dataset` object: Outputs of `Dataset` object must have
                same constraints as below.
              * features: A `tf.Tensor` or a dictionary of string feature name to
                `Tensor`. features are consumed by `model_fn`. They should satisfy
                the expectation of `model_fn` from inputs.
              * A tuple, in which case the first item is extracted as features.

        :param checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, prediction is run with newly initialized `Variables`
            instead of ones restored from checkpoint.


        Return:
          Evaluated values of `predictions` tensors.

        """
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

                    rdd = result.get_prediction_data()

                    results = tfnet.predict(rdd, result.batch_per_thread)

                    # If predictions is a dict, add back the keys and results is a dict as well.
                    if isinstance(spec.predictions, dict):
                        # Given a list of outputs; return a dict of outputs.
                        def zip_key(outs, keys):
                            assert len(outs) == len(keys)
                            res_dict = {}
                            for out, key in zip(outs, keys):
                                res_dict[key] = out
                            return res_dict

                        pred_keys = sorted(spec.predictions.keys())
                        results = results.map(lambda res: zip_key(res, pred_keys))
                    return results

        return list(self.estimator.predict(input_fn, checkpoint_path=checkpoint_path))

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
