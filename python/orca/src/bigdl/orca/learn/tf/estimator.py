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

from pyspark.sql.dataframe import DataFrame

from bigdl.optim.optimizer import MaxEpoch
from zoo.tfpark.utils import evaluate_metrics
from zoo.tfpark import TFOptimizer, TFNet, ZooOptimizer
from zoo.tfpark import KerasModel
from zoo.util import nest
from zoo.orca.learn.tf.utils import to_dataset, convert_predict_to_dataframe


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   clip_norm=None, clip_value=None,
                   metrics=None, updates=None,
                   sess=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  clip_norm=clip_norm,
                                  clip_value=clip_value,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir
                                  )

    @staticmethod
    def from_keras(keras_model, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        return TFKerasWrapper(keras_model, model_dir)


class TFOptimizerWrapper(Estimator):

    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer, clip_norm, clip_value,
                 metrics,
                 updates, sess,
                 model_dir
                 ):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        if optimizer is not None:
            assert isinstance(optimizer, tf.train.Optimizer), \
                "optimizer is of type {}, ".format(type(self.optimizer)) + \
                "it should be an instance of tf.train.Optimizer"
            self.optimizer = ZooOptimizer(optimizer)
            if clip_norm or clip_value:
                gvs = self.optimizer.compute_gradients(self.loss)
                if clip_norm:
                    gvs = [(tf.clip_by_norm(g_v[0], clip_norm), g_v[1]) for g_v in gvs]
                if clip_value:
                    if isinstance(clip_value, tuple):
                        assert len(clip_value) == 2 and clip_value[0] < clip_value[1], \
                            "clip value should be (clip_min, clip_max)"
                        gvs = [(tf.clip_by_value(g_v[0], clip_value[0], clip_value[1]), g_v[1])
                               for g_v in gvs]
                    if isinstance(clip_value, (int, float)):
                        assert clip_value > 0, "clip value should be larger than 0"
                        gvs = [(tf.clip_by_value(g_v[0], -clip_value, clip_value), g_v[1])
                               for g_v in gvs]
                    else:
                        raise Exception("clip_value should be a tuple or one number")
                self.train_op = self.optimizer.apply_gradients(gvs)
            else:
                self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.optimizer = None
            self.train_op = None
        self.metrics = metrics
        self.updates = updates
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.model_dir = model_dir

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            session_config=None,
            feed_dict=None
            ):

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it should not be None in training"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True
                             )

        if feed_dict is not None:
            tensor_with_value = {key: (value, value) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        optimizer = TFOptimizer.from_train_op(
            train_op=self.train_op,
            loss=self.loss,
            inputs=self.inputs,
            labels=self.labels,
            dataset=dataset,
            metrics=self.metrics,
            updates=self.updates, sess=self.sess,
            tensor_with_value=tensor_with_value,
            session_config=session_config,
            model_dir=self.model_dir)

        optimizer.optimize(end_trigger=MaxEpoch(epochs))
        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False
                ):

        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"
        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        predicted_rdd = tfnet.predict(dataset)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=32,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False
                 ):

        assert self.metrics is not None, \
            "metrics is None, it should not be None in evaluate"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_labels = nest.flatten(self.labels)

        return evaluate_metrics(flat_inputs + flat_labels,
                                sess=self.sess,
                                dataset=dataset, metrics=self.metrics)


class TFKerasWrapper(Estimator):

    def __init__(self, keras_model, model_dir):
        self.model = KerasModel(keras_model, model_dir)

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            session_config=None
            ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True
                             )

        self.model.fit(dataset, batch_size=batch_size, epochs=epochs,
                       session_config=session_config
                       )
        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False
                ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False
                             )

        predicted_rdd = self.model.predict(dataset, batch_size)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=4,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False
                 ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False
                             )

        return self.model.evaluate(dataset, batch_per_thread=batch_size)
