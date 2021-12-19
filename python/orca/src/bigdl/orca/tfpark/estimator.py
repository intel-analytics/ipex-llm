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

from bigdl.dllib.optim.optimizer import MaxIteration, Loss, TreeNNAccuracy

from bigdl.orca.tfpark.utils import evaluate_string_metrics
from bigdl.dllib.keras import metrics
from bigdl.orca.tfpark.tfnet import TFNet
from bigdl.orca.tfpark.tf_optimizer import TFOptimizer
from bigdl.orca.tfpark.tf_dataset import TFDataset

from bigdl.dllib.utils import nest
import six
import os


class TFEstimator(object):

    def __init__(self, estimator):
        """
        :param estimator: a tf.estimator.Estimator, in the estimator's model_fn,
        ZooOptimizer must be used and only ZooOptimizer should be used to derive
        the train_op.
        """

        self.estimator = estimator
        self._model_fn = self.estimator.model_fn
        self._model_dir = self.estimator.model_dir
        self.config = self.estimator.config
        self.params = self.estimator.params
        self.tf_optimizer = None

    @classmethod
    def from_model_fn(cls, model_fn, model_dir=None, config=None,
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
                `tf.estimator.EstimatorSpec`

            For the train_op in tf.estimator.EstimatorSpec, it derive from and only from
            `zoo.tfpark.zoo_optimizer.ZooOptimizer`
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
        import tensorflow as tf
        estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config,
                                           params=params, warm_start_from=warm_start_from)
        return cls(estimator)

    def _call_model_fn(self, features, labels, mode, config):
        from tensorflow.python.util import function_utils
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

        return model_fn_results

    def train(self, input_fn, steps=None, session_config=None):
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
        import tensorflow as tf

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
                latest_checkpoint = self.estimator.latest_checkpoint()

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    if latest_checkpoint:
                        saver.restore(sess, latest_checkpoint)
                    else:
                        sess.run(tf.global_variables_initializer())

                    zoo_ckpt_path = os.path.join(self._model_dir, "analytics-zoo")

                    opt = TFOptimizer.from_train_op(spec.train_op,
                                                    spec.loss,
                                                    sess=sess,
                                                    dataset=result,
                                                    model_dir=zoo_ckpt_path,
                                                    session_config=session_config)

                    start_step = sess.run(global_step_tensor)
                    opt.optimize(MaxIteration(steps))
                    final_step = sess.run(global_step_tensor)
                    if final_step == start_step:
                        # user does not increase global step
                        sess.run(assign_step, feed_dict={add_step_input: steps})
                        final_step += steps

                    model_path = os.path.join(self._model_dir, "model")
                    saver.save(sess, model_path, global_step=final_step)
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
        from tensorflow_estimator.python.estimator.canned import prediction_keys
        import tensorflow as tf
        with tf.Graph().as_default() as g:
            result = self.estimator._call_input_fn(input_fn, tf.estimator.ModeKeys.EVAL)
            if isinstance(result, TFDataset):
                spec = self._call_model_fn(result.feature_tensors,
                                           result.label_tensors,
                                           tf.estimator.ModeKeys.EVAL,
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
                    if isinstance(spec.predictions, dict):
                        if "mae" in eval_methods:
                            key = prediction_keys.PredictionKeys.PREDICTIONS
                            msg = "{} is required for evaluating mse,".format(key) + \
                                  " please add it in your model_fn predictions"
                            assert key in spec.prediction, msg
                            outputs = [
                                spec.predictions[prediction_keys.PredictionKeys.PREDICTIONS]]
                        else:
                            key = prediction_keys.PredictionKeys.LOGITS
                            msg = "{} is required in for evaluating,".format(key) + \
                                  " please add it in your model_fn predictions"
                            assert key in spec.predictions, msg
                            outputs = [
                                spec.predictions[prediction_keys.PredictionKeys.LOGITS]]
                    else:
                        outputs = nest.flatten(spec.predictions)
                        if len(outputs) > 1:
                            raise Exception("Evaluate on more than one output is not " +
                                            "supported now")

                    all_inputs = result._original_tensors
                    if isinstance(all_inputs, tuple) and len(all_inputs) == 2:
                        targets = nest.flatten(all_inputs[1])
                    else:
                        targets = None
                    return evaluate_string_metrics(sess=sess,
                                                   string_metrics=eval_methods,
                                                   dataset=result,
                                                   inputs=nest.flatten(all_inputs),
                                                   targets=targets,
                                                   outputs=outputs,
                                                   loss=spec.loss)

        return self.estimator.evaluate(input_fn, steps, checkpoint_path=checkpoint_path)

    def predict(self, input_fn, predict_keys=None, checkpoint_path=None):
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
        import tensorflow as tf

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
                    if isinstance(spec.predictions, dict) and predict_keys is not None:
                        outputs = [spec.predictions[key] for key in predict_keys]
                    else:
                        outputs = nest.flatten(spec.predictions)
                    tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)
                    predictions = tfnet.predict(result.get_prediction_data(), mini_batch=True)

                    # If predictions is a dict, add back the keys and results is a dict as well.
                    if isinstance(spec.predictions, dict):
                        # Given a list of outputs; return a dict of outputs.
                        def zip_key(outs, keys):
                            if isinstance(outs, list):
                                error_msg = "output length is " \
                                    + "{} but keys length is {}".format(len(outs), len(keys))
                                assert len(outs) == len(keys), error_msg
                            else:
                                outs = [outs]
                            res_dict = {}
                            for out, key in zip(outs, keys):
                                res_dict[key] = out
                            return res_dict

                        pred_keys = sorted(spec.predictions.keys()) if not predict_keys \
                            else predict_keys
                        predictions = predictions.map(lambda res: zip_key(res, pred_keys))
                    return predictions

        return list(self.estimator.predict(input_fn, checkpoint_path=checkpoint_path))

    @staticmethod
    def _to_bigdl_metric(metric):
        metric = metric.lower()
        if metric == "accuracy" or metric == "acc":
            return metrics.Accuracy()
        elif metric == "top5accuracy" or metric == "top5acc":
            return metrics.Top5Accuracy()
        elif metric == "mae":
            from bigdl.dllib.optim.optimizer import MAE
            return MAE()
        elif metric == "auc":
            return metrics.AUC()
        elif metric == "treennaccuracy":
            return TreeNNAccuracy()
        else:
            raise TypeError("Unsupported metric: %s" % metric)
