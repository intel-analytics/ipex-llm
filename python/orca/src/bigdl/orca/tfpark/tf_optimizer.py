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

import json
import logging
import os
import sys
import tempfile

from bigdl.dllib.nn.criterion import Criterion
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.optim.optimizer import MaxEpoch, EveryEpoch
from bigdl.dllib.utils.common import to_list, JavaValue

from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.feature.common import FeatureSet
from bigdl.dllib.keras.engine.topology import to_bigdl_metric, Loss, OptimMethod
from bigdl.dllib.net.utils import find_placeholders, to_bigdl_optim_method, find_tensors
from bigdl.dllib.estimator.estimator import Estimator
from bigdl.dllib.utils import nest
from bigdl.dllib.utils.triggers import EveryEpoch as ZEveryEpoch
from bigdl.dllib.utils.triggers import ZooTrigger
from bigdl.orca.tfpark.tf_dataset import TFNdarrayDataset, check_data_compatible
from bigdl.orca.tfpark.tf_dataset import _standarize_feature_label_dataset
from bigdl.dllib.utils.log4Error import *

if sys.version >= '3':
    long = int
    unicode = str


class IdentityCriterion(Criterion):
    def __init__(self):
        super(IdentityCriterion, self).__init__(None, "float")


class TFValidationMethod(JavaValue):
    def __init__(self, val_method, name, output_indices, label_indices):
        self.name = name
        self.val_method = val_method
        JavaValue.__init__(self, None, "float",
                           val_method, name, output_indices, label_indices)


class StatelessMetric(JavaValue):
    def __init__(self, metric_name, idx, count_idx):
        self.name = metric_name
        self.idx = idx
        self.count_idx = count_idx
        JavaValue.__init__(self, None, "float", metric_name, idx, count_idx)


class BigDLMetric(object):
    def __init__(self, val_method, outputs, labels):
        self.val_method = val_method
        self.outputs = outputs
        self.labels = labels


class TFTrainingHelper(Layer):
    def __init__(self, path, config_proto, saver, meta, sess):
        self.saver = saver
        self.meta = meta
        self.export_dir = path
        self.sess = sess

        if config_proto is not None:
            import tensorflow as tf
            invalidInputError(isinstance(config_proto, tf.ConfigProto),
                              "session_config should be a tf.ConfigProto")
            config_proto.use_per_session_threads = True
            byte_arr = bytearray(config_proto.SerializeToString())
        else:
            byte_arr = None

        super(TFTrainingHelper, self).__init__(None, "float", path, byte_arr)

    def save_checkpoint(self):
        callZooFunc(self.bigdl_type, "saveCheckpoint",
                    self.value)

    def get_weights_to_python(self):
        self.save_checkpoint()
        self.saver.restore(self.sess, os.path.join(self.export_dir, "model"))

    def load_checkpoint(self, path):
        callZooFunc(self.bigdl_type, "loadZooCheckpoint", self.value, path)
        self.get_weights_to_python()


def _to_operation_name(name):
    return name.split(":")[0]


def _to_floats(vs):
    return [float(v) for v in vs]


class TFModel(object):
    def __init__(self, training_helper_layer, criterion, val_methods):

        self.training_helper_layer = training_helper_layer
        self.criterion = criterion
        self.val_methods = val_methods

    @staticmethod
    def _expand_inputs(inputs, tensors_with_value, loss):
        additional_inputs = []
        additional_values = []
        inputs = nest.flatten(inputs)
        names = set([i.name for i in inputs])

        if tensors_with_value:
            for t, v in tensors_with_value.items():
                if t.name in names:
                    msg = f"tensor {t} already in inputs, cannot put it in tensor_with_value"
                    invalidInputError(False, msg)
                additional_inputs.append(t)
                additional_values.append(v)

        return inputs, additional_inputs, additional_values

    @staticmethod
    def _process_session_config(session_config):
        import tensorflow as tf
        if session_config is not None:
            invalidInputError(isinstance(session_config, tf.ConfigProto),
                              "session_config should be a tf.ConfigProto")
            session_config.use_per_session_threads = True
        return session_config

    @staticmethod
    def _process_grads(graph, grads):

        with graph.as_default():
            from bigdl.dllib.utils.tf import process_grad
            grads = [process_grad(grad) for grad in grads]
        return grads

    @staticmethod
    def _process_metrics(graph, metrics, real_batch_size):
        import tensorflow as tf
        outputs = [real_batch_size]
        val_methods = None
        if metrics is not None:
            idx = 1
            val_methods = []
            for metric_name in metrics:
                metric = metrics[metric_name]
                if tf.is_numeric_tensor(metric):
                    outputs.append(metric)
                    val_methods.append(StatelessMetric(metric_name, idx, 0))
                    idx += 1
                else:
                    outputs += metric.outputs
                    with graph.as_default():
                        val_labels = [tf.identity(v) for v in metric.labels]
                    outputs += val_labels
                    method = TFValidationMethod(metric.val_method,
                                                metric_name,
                                                list(range(idx, idx + len(metric.outputs))),
                                                list(range(idx + len(metric.outputs),
                                                           idx + len(metric.outputs)
                                                           + len(val_labels))))
                    val_methods.append(method)
                    idx += len(metric.outputs) + len(val_labels)

        outputs = [tf.to_float(output) for output in outputs]
        return outputs, val_methods

    @staticmethod
    def _process_variables(graph, variables, updates):
        import tensorflow as tf
        all_trainable_variables = variables

        name2idx = dict([(v.name, idx) for idx, v in enumerate(all_trainable_variables)])

        all_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        update_ops = graph.get_collection(tf.GraphKeys.UPDATE_OPS)

        if updates is not None:
            update_ops += updates

        trainable_variables = [0] * len(all_trainable_variables)
        trainable_assigns = [0] * len(all_trainable_variables)
        trainable_variable_placeholders = [0] * len(all_trainable_variables)
        extra_variables = []
        extra_variable_assigns = []
        extra_variable_assign_placeholders = []
        for v in all_variables:
            p = tf.placeholder(dtype=v.dtype, shape=v.shape)
            a = tf.assign(v, p)

            # special treatment for ResourceVariable
            if v.op.type == "VarHandleOp":
                v_float_value = tf.to_float(v.read_value())
            else:
                v_float_value = tf.to_float(v)

            if v.name in name2idx:
                trainable_variables[name2idx[v.name]] = v_float_value
                trainable_assigns[name2idx[v.name]] = a
                trainable_variable_placeholders[name2idx[v.name]] = p
            else:
                extra_variables.append(v_float_value)
                extra_variable_assigns.append(a)
                extra_variable_assign_placeholders.append(p)

        extra_variable_assign = tf.group(*extra_variable_assigns)
        trainable_assign = tf.group(*trainable_assigns)
        update_op = tf.group(update_ops)

        return trainable_variables, trainable_variable_placeholders, trainable_assign, \
            extra_variables, extra_variable_assign_placeholders, \
            extra_variable_assign, update_op

    @staticmethod
    def _save_to_dir(folder, sess, graph,
                     metric_tensors,
                     batch_size_tensor,
                     loss_tensor, inputs, labels, predictions,
                     trainable_variables,
                     trainable_variable_placeholders,
                     trainable_assign,
                     extra_variables,
                     extra_variable_assign_placeholders,
                     extra_variable_assign,
                     grads, update_op, train_op,
                     additional_inputs,
                     additional_values):
        import tensorflow as tf
        from tensorflow import gfile
        saver = tf.train.Saver()
        if not os.path.isdir(folder):
            os.makedirs(folder)
        saver.save(sess, os.path.join(folder, "model"), write_meta_graph=False)

        meta = {
            "inputs": [i.name for i in inputs],
            "input_types": [i.dtype.as_datatype_enum for i in inputs],
            "additional_inputs": [i.name for i in additional_inputs],
            "additional_input_types": [i.dtype.as_datatype_enum for i in additional_inputs],
            "labels": [l.name for l in labels],
            "label_types": [i.dtype.as_datatype_enum for i in labels],
            "predictions": [t.name for t in predictions] if predictions else [],
            "metric_tensors": [t.name for t in metric_tensors],
            "batch_size_tensor": batch_size_tensor.name,
            "loss_tensor": loss_tensor.name,
            "variables": [v.name for v in trainable_variables],
            "variable_types": [v.dtype.as_datatype_enum for v in trainable_variable_placeholders],
            "variable_assign_placeholders": [v.name for v in trainable_variable_placeholders],
            "assign_variable_op": trainable_assign.name,
            "extra_variables": [v.name for v in extra_variables],
            "extra_variable_types": [v.dtype.as_datatype_enum for v
                                     in extra_variable_assign_placeholders],
            "extra_variable_assign_placeholders": [p.name for p in
                                                   extra_variable_assign_placeholders],
            "assign_extra_variable_op": extra_variable_assign.name,
            "grad_variables": [g.name for g in grads],
            "update_op": update_op.name,
            "restore_op": saver.saver_def.restore_op_name,
            "restore_path_placeholder": saver.saver_def.filename_tensor_name,
            "save_op": _to_operation_name(saver.saver_def.save_tensor_name),
            "save_path_placeholder": saver.saver_def.filename_tensor_name,
            "default_tensor_value": [_to_floats(v) for v in additional_values],
            "init_op": tf.tables_initializer().name
        }

        if train_op is not None:
            meta["train_op"] = train_op.name

        with open(os.path.join(folder, "training_meta.json"), "w") as f:
            f.write(json.dumps(meta))

        with gfile.GFile(os.path.join(folder, "model.meta"), "wb") as f:
            f.write(graph.as_graph_def().SerializeToString())

        return meta, saver

    @staticmethod
    def export(model_dir, loss_tensor, sess, inputs, labels, predictions, grads, variables, graph,
               tensors_with_value, metrics, updates, train_op=None):
        import tensorflow as tf
        with graph.as_default():
            batch_size_tensor = tf.to_float(tf.shape(inputs[0])[0])
        inputs, additional_inputs, additional_values = \
            TFModel._expand_inputs(inputs, tensors_with_value, loss_tensor)
        metric_tensors, val_methods = TFModel._process_metrics(graph, metrics, batch_size_tensor)
        grads = TFModel._process_grads(graph, grads)

        trainable_variables, trainable_variable_placeholders, trainable_assign, \
            extra_variables, extra_variable_assign_placeholders, \
            extra_variable_assign, update_op = \
            TFModel._process_variables(graph, variables, updates)

        meta, saver = \
            TFModel._save_to_dir(model_dir, sess, graph,
                                 metric_tensors,
                                 batch_size_tensor,
                                 loss_tensor, inputs, labels, predictions,
                                 trainable_variables,
                                 trainable_variable_placeholders,
                                 trainable_assign,
                                 extra_variables,
                                 extra_variable_assign_placeholders,
                                 extra_variable_assign,
                                 grads, update_op, train_op,
                                 additional_inputs,
                                 additional_values)
        return meta, saver, val_methods

    @staticmethod
    def create(loss_tensor, sess, inputs, labels, predictions, grads, variables, graph,
               tensors_with_value, session_config, metrics, updates,
               model_dir, train_op=None):

        if model_dir is None:
            model_dir = tempfile.mkdtemp()
        else:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        meta, saver, val_methods = TFModel.export(model_dir, loss_tensor, sess,
                                                  inputs, labels, predictions, grads, variables,
                                                  graph, tensors_with_value, metrics, updates,
                                                  train_op)

        training_helper_layer = TFTrainingHelper(model_dir,
                                                 session_config, saver, meta, sess)

        criterion = IdentityCriterion()

        return TFModel(training_helper_layer, criterion, val_methods)


class TFOptimizer:
    def __init__(self, tf_model, optim_method,
                 sess=None, dataset=None,
                 clip_norm=None, clip_value=None,
                 model_dir=None):
        """
        TFOptimizer is used for distributed training of TensorFlow
        on Spark/BigDL.

        Note that if grads and variables are not None, then they need to be sorted by name
        if you want to use multiple optimization methods for a TensorFlow model according to
        variable names.

        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as
        bigdl.dllib.optim.optimizer.Adam
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        """

        self.optim_method = optim_method
        self.sess = sess
        self.dataset = dataset

        self.clip_norm = clip_norm
        if clip_value is not None and not isinstance(clip_value, tuple):
            invalidInputError(False,
                              "The clip_value argument should be a tuple (min_value, max_value)")
        self.clip_constant = clip_value

        if self.dataset.batch_size <= 0:
            invalidInputError(False,
                              "You should set batch_size instead of batch_per_thread for training")

        self.model_dir = model_dir

        self.tf_model = tf_model

        batch_size = self.dataset.batch_size

        self.train_data = self.dataset.get_training_data()
        self.val_data = self.dataset.get_validation_data()

        self.batch_size = batch_size

        self.estimator = Estimator(self.tf_model.training_helper_layer,
                                   self.optim_method,
                                   self.model_dir)

        if self.clip_norm:
            self.estimator.set_l2_norm_gradient_clipping(self.clip_norm)
        if self.clip_constant:
            min_value, max_value = self.clip_constant
            self.estimator.set_constant_gradient_clipping(min_value, max_value)

    def load_checkpoint(self, path, version):
        # todo make version optional
        model_path = os.path.join(path, "model.{}".format(version))
        optim_method_path = os.path.join(path, "optimMethod-TFParkTraining.{}".format(version))
        self.tf_model.training_helper_layer.load_checkpoint(model_path)
        self.optim_method = OptimMethod.load(optim_method_path)
        self.estimator = Estimator(self.tf_model.training_helper_layer,
                                   self.optim_method,
                                   self.model_dir)
        if self.clip_norm:
            self.estimator.set_l2_norm_gradient_clipping(self.clip_norm)
        if self.clip_constant:
            min_value, max_value = self.clip_constant
            self.estimator.set_constant_gradient_clipping(min_value, max_value)

    @staticmethod
    def _get_or_create_session(session):
        import tensorflow as tf
        if session is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        else:
            sess = session
        return sess

    @staticmethod
    def _get_dataset_from_loss(loss):
        import tensorflow as tf
        all_required_inputs = find_placeholders([loss])
        dataset = tf.get_collection(all_required_inputs[0].name)[0]
        return dataset

    @staticmethod
    def _get_vars_grads(loss):
        import tensorflow as tf
        grads_vars = tf.train.GradientDescentOptimizer(0).compute_gradients(loss)
        grads_vars.sort(key=lambda grad_var: grad_var[1].name)
        variables = []
        grads = []
        for (grad, var) in grads_vars:
            if grad is not None:
                variables.append(var)
                grads.append(grad)
        return grads, variables

    @staticmethod
    def _get_vars_grads_from_train_op(train_op):
        def predicate(t):
            return t.name.split("/")[-1].startswith("zoo_identity_op_for_grad")
        grads = find_tensors([train_op], predicate)
        grad_ops = [grad.op for grad in grads]
        variables = []
        for grad in grad_ops:
            var = list(grad.control_inputs)[0]
            if var.name == "VarHandleOp":
                variables.append(var)
            else:
                variables.append(list(var.outputs)[0])
        # variables = [grad.op.control_inputs[0].outputs[0] for grad in grads]
        return grads, variables

    @classmethod
    def from_train_op(cls, train_op, loss, *, inputs=None, labels=None, metrics=None, updates=None,
                      sess=None, dataset=None, tensor_with_value=None, session_config=None,
                      model_dir=None):

        sess = TFOptimizer._get_or_create_session(sess)
        grads, variables = TFOptimizer._get_vars_grads_from_train_op(train_op)
        if dataset is None:
            dataset = TFOptimizer._get_dataset_from_loss(loss)
        _ = dataset.tensors  # trigger create tensors if not available
        dataset_inputs = dataset._original_tensors
        if isinstance(dataset_inputs, tuple) and len(dataset_inputs) == 2:
            if inputs is None:
                inputs = dataset_inputs[0]

            if labels is None:
                labels = dataset_inputs[1]
        else:
            if inputs is None:
                inputs = dataset_inputs

            if labels is None:
                labels = []

        inputs = nest.flatten(inputs)
        labels = nest.flatten(labels)
        from bigdl.orca.tfpark.zoo_optimizer import FakeOptimMethod
        return TFOptimizer._from_grads(loss=loss, sess=sess, inputs=inputs, labels=labels,
                                       grads=grads,
                                       variables=variables, dataset=dataset, metrics=metrics,
                                       tensor_with_value=tensor_with_value,
                                       optim_method=FakeOptimMethod(),
                                       session_config=session_config, updates=updates,
                                       model_dir=model_dir, train_op=train_op)

    @classmethod
    def _from_grads(cls, loss, sess, inputs, labels, grads, variables, dataset, optim_method=None,
                    clip_norm=None, clip_value=None,
                    metrics=None, tensor_with_value=None, session_config=None,
                    model_dir=None, updates=None, train_op=None):
        graph = loss.graph
        if metrics is None:
            metrics = {}

        tf_model = TFModel.create(loss, sess, inputs, labels, [], grads, variables, graph,
                                  tensor_with_value, session_config, metrics,
                                  updates, model_dir=None, train_op=train_op)
        return cls(tf_model, optim_method, sess=sess, dataset=dataset,
                   clip_norm=clip_norm, clip_value=clip_value, model_dir=model_dir)

    @classmethod
    def from_loss(cls, loss, optim_method, session=None, inputs=None, dataset=None,
                  val_outputs=None, val_labels=None, val_method=None,
                  clip_norm=None, clip_value=None, metrics=None,
                  tensor_with_value=None, session_config=None, model_dir=None, updates=None):
        """
        Create a TFOptimizer from a TensorFlow loss tensor.
        The loss tensor must come from a TensorFlow graph that only takes TFDataset.tensors and
        the tensors in `tensor_with_value` as inputs.
        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as
        bigdl.dllib.optim.optimizer.Adam
        :param session: the current TensorFlow Session, if you want to used a pre-trained model,
        you should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        :param val_outputs: the validation output TensorFlow tensor to be used by val_methods
        :param val_labels: the validation label TensorFlow tensor to be used by val_methods
        :param val_method: the BigDL val_method(s) to be used.
        :param clip_norm: float >= 0. Gradients will be clipped when their L2 norm exceeds
        this value.
        :param clip_value: float >= 0. Gradients will be clipped when their absolute value
        exceeds this value.
        :param metrics: a dictionary. The key should be a string representing the metric's name
        and the value should be the corresponding TensorFlow tensor, which should be a scalar.
        :param tensor_with_value: a dictionary. The key is TensorFlow tensor, usually a
        placeholder, the value of the dictionary is a tuple of two elements. The first one of
        the tuple is the value to feed to the tensor in training phase and the second one
        is the value to feed to the tensor in validation phase.
        :return: a TFOptimizer
        """
        sess = TFOptimizer._get_or_create_session(session)
        grads, variables = TFOptimizer._get_vars_grads(loss)

        if dataset is None and inputs is None:
            dataset = TFOptimizer._get_dataset_from_loss(loss)
            inputs = dataset._original_tensors
        else:
            if inputs is None:
                invalidInputError(False, "please specify inputs")
            _ = dataset.tensors  # trigger creating placeholders

        if isinstance(inputs, tuple) and len(inputs) == 2:
            inputs, labels = inputs
        else:
            labels = []

        inputs = nest.flatten(inputs)
        labels = nest.flatten(labels)

        if clip_value is not None:
            if isinstance(clip_value, float) or isinstance(clip_value, int):
                if clip_value <= 0:
                    ValueError("The clip_value argument should be positive number")
                clip_value = (-float(clip_value), float(clip_value))

            if not isinstance(clip_value, tuple):
                invalidInputError(False,
                                  "The clip_value argument should be" +
                                  " a positive float/int which clips to" +
                                  " (-clip_value, clip_value); " +
                                  "or a tuple which clips to (min_value, max_value)")

        if val_method is not None:
            val_methods = to_list(val_method)
            if metrics is None:
                metrics = {}

            for i, method in enumerate(val_methods):
                metrics['bigdl_metric_' + str(i)] = BigDLMetric(method, val_outputs, val_labels)

        return TFOptimizer._from_grads(loss, sess, inputs, labels, grads, variables, dataset,
                                       optim_method, clip_norm, clip_value,
                                       metrics, tensor_with_value, session_config,
                                       model_dir, updates)

    @staticmethod
    def export_training_model(export_dir, loss, sess, inputs, labels=None, predictions=None,
                              metrics=None, tensor_with_value=None, updates=None):

        grads, variables = TFOptimizer._get_vars_grads(loss)

        TFModel.export(export_dir, loss, sess, inputs, labels, predictions, grads, variables,
                       loss.graph, tensor_with_value, metrics, updates)
        logging.info("Exported TensorFlow model in {} for training".format(export_dir))

    @staticmethod
    def _shape_match(model_shape, dataset_shape):

        for i in range(len(dataset_shape)):
            if dataset_shape[i].value is None:
                return model_shape[i].value is None
            else:
                return dataset_shape[i].value == model_shape[i].value or \
                    model_shape[i].value is None

    @classmethod
    def from_keras(cls, keras_model, dataset,
                   session_config=None, model_dir=None, metrics=None, optimizer=None):
        """
        Create a TFOptimizer from a tensorflow.keras model. The model must be compiled.
        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param dataset: a TFDataset
        :return:
        """
        import tensorflow.keras.backend as K

        model_inputs = keras_model.inputs

        if hasattr(keras_model, "targets"):
            model_targets = keras_model.targets
        else:
            model_targets = keras_model._targets

        # target can be None if loss is None
        model_targets = list(filter(lambda x: x is not None, model_targets))

        check_data_compatible(dataset, keras_model, mode="train")
        # standarize feature, labels to support keras model
        if isinstance(dataset, TFNdarrayDataset):
            dataset = _standarize_feature_label_dataset(dataset, keras_model)

        flatten_inputs = nest.flatten(dataset.feature_tensors)
        invalidInputError(len(model_inputs) == len(flatten_inputs),
                          "the keras model and TFDataset should have the same number of tensors"
                          " keras model has {} inputs while TFDataset has {} "
                          "inputs".format(len(model_inputs), len(flatten_inputs)))
        for i in range(len(flatten_inputs)):
            if not TFOptimizer._shape_match(model_inputs[i].shape, flatten_inputs[i].shape):
                invalidInputError(False,
                                  ("The {}th input in keras model {}"
                                   " does not match the TFDataset"
                                   "input {}").format(i,
                                                      model_inputs[i],
                                                      flatten_inputs[i]))

        flatten_targets = nest.flatten(dataset.label_tensors)
        invalidInputError(len(model_targets) == len(flatten_targets),
                          "the keras model and TFDataset should have the same number"
                          " of tensors keras model has {} targets while TFDataset has"
                          " {} labels".format(len(model_targets), len(flatten_inputs)))
        # todo check targets shape, currently checking target shape will
        # cause too much false alarm.

        loss = keras_model.total_loss
        variables = keras_model._collected_trainable_weights
        variables.sort(key=lambda variable: variable.name)
        keras_optimizer = keras_model.optimizer

        from bigdl.orca.tfpark.zoo_optimizer import get_gradients_for_keras
        grads = get_gradients_for_keras(keras_optimizer, loss, variables)
        grads_and_vars = list(zip(grads, variables))
        import tensorflow.python.keras.optimizers as koptimizers
        if isinstance(keras_optimizer, koptimizers.TFOptimizer):
            # work around keras TFOptimzier bug
            train_op = keras_optimizer.optimizer.apply_gradients(grads_and_vars)
        else:
            train_op = keras_optimizer.apply_gradients(grads_and_vars)

        sess = K.get_session()

        if keras_model.metrics and (dataset.get_validation_data() is not None):
            if isinstance(keras_model.metrics, dict):
                invalidInputError(False,
                                  "different metrics for different outputs are not"
                                  " supported right now")

            if len(keras_model.outputs) > 1:
                if not all([name.endswith("loss") for name in keras_model.metrics_names]):
                    invalidInputError(False,
                                      "metrics (except loss) for multi-head model is not"
                                      " supported")
                else:
                    bigdl_val_methods = [Loss()]
                    val_outputs = keras_model.outputs
                    val_labels = model_targets
            else:
                bigdl_val_methods = \
                    [to_bigdl_metric(m, keras_model.loss) for m in keras_model.metrics_names]
                val_outputs = keras_model.outputs
                val_labels = model_targets
        else:
            val_outputs = None
            val_labels = None
            bigdl_val_methods = None

        tensor_with_value = {
            K.learning_phase(): [True, False]
        }

        updates = []

        updates += keras_model.get_updates_for(None)
        # Conditional updates relevant to this model
        updates += keras_model.get_updates_for(keras_model.inputs)

        if bigdl_val_methods is not None:
            val_methods = to_list(bigdl_val_methods)
            bigdl_metrics = {}
            for i, method in enumerate(val_methods):
                bigdl_metrics['bigdl_metric_' + str(i)] = BigDLMetric(method,
                                                                      val_outputs,
                                                                      val_labels)
            if metrics is None:
                metrics = bigdl_metrics
            else:
                metrics.update(bigdl_metrics)

        if optimizer is not None:
            clip_norm = None
            clip_value = None
            if hasattr(keras_optimizer, 'clipnorm'):
                clip_norm = keras_optimizer.clipnorm
            if hasattr(keras_optimizer, 'clipvalue'):
                clip_value = (-keras_optimizer.clipvalue, keras_optimizer.clipvalue)
            tf_model = TFModel.create(loss, sess, model_inputs, model_targets, keras_model.outputs,
                                      grads, variables, loss.graph,
                                      tensor_with_value, session_config, metrics,
                                      updates, model_dir=None)

            return cls(tf_model, optimizer, sess=sess, dataset=dataset,
                       clip_norm=clip_norm, clip_value=clip_value, model_dir=model_dir)

        return cls.from_train_op(train_op, loss, inputs=model_inputs, labels=model_targets,
                                 metrics=metrics, updates=updates, sess=sess, dataset=dataset,
                                 tensor_with_value=tensor_with_value, session_config=session_config,
                                 model_dir=model_dir)

    def set_constant_gradient_clipping(self, min_value, max_value):
        """
        Configure constant clipping settings.

        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        self.estimator.set_constant_gradient_clipping(min_value, max_value)

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        """
        Configure L2 norm clipping settings.
        :param clip_norm: gradient L2-Norm threshold
        """
        self.estimator.set_l2_norm_gradient_clipping(clip_norm)

    def optimize(self, end_trigger=None, checkpoint_trigger=None):
        """
        Run the training loop of the this optimizer
        :param end_trigger: BigDL's Trigger to indicate when to stop the training.
        :param checkpoint_trigger: When to save a checkpoint and evaluate model.
        """
        if end_trigger is None:
            end_trigger = MaxEpoch(1)

        if checkpoint_trigger is None:
            checkpoint_trigger = EveryEpoch()

        if isinstance(self.train_data, FeatureSet):
            if self.train_data.value.getNumOfSlice() != 1:
                if isinstance(checkpoint_trigger, EveryEpoch):
                    checkpoint_trigger = ZEveryEpoch()
                elif not isinstance(checkpoint_trigger, ZooTrigger):
                    invalidInputError(False,
                                      "Please use a trigger defined in bigdl.dllib.utils.triggers")

        if self.tf_model.val_methods and self.val_data is not None:
            self.estimator.train_minibatch(train_set=self.train_data,
                                           criterion=self.tf_model.criterion,
                                           end_trigger=end_trigger,
                                           checkpoint_trigger=checkpoint_trigger,
                                           validation_set=self.val_data,
                                           validation_method=self.tf_model.val_methods)
        else:
            self.estimator.train_minibatch(train_set=self.train_data,
                                           criterion=self.tf_model.criterion,
                                           end_trigger=end_trigger,
                                           checkpoint_trigger=checkpoint_trigger)

        self.tf_model.training_helper_layer.get_weights_to_python()
