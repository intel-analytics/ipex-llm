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
from bigdl.optim.optimizer import MaxIteration, SGD

from zoo.tfpark.utils import evaluate_metrics
from zoo.orca.data import SparkXShards
from zoo.orca.data.tf.data import Dataset, TFDataDataset2
from zoo.tfpark import TFOptimizer, TFNet, ZooOptimizer
import tensorflow as tf

from zoo.tfpark.tf_dataset import TFDataset
from zoo.tfpark.tf_optimizer import TFModel
from zoo.util import nest


class Estimator(object):
    def fit(self, data, steps, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   metrics=None, updates=None,
                   sess=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir)


def _xshards_to_tf_dataset(data_shard,
                           batch_size=-1, batch_per_thread=-1,
                           validation_data_shard=None):
    # todo data_shard.head ?
    import numpy as np

    def check_data_type_and_to_list(data):
        result = {}
        assert isinstance(data, dict), "each shard should be an dict"
        assert "x" in data, "key x should in each shard"
        x = data["x"]
        if isinstance(x, np.ndarray):
            new_x = [x]
        elif isinstance(x, tuple) and all([isinstance(xi, np.ndarray) for xi in x]):
            new_x = x
        else:
            raise ValueError("value of x should be a ndarray or a tuple of ndarrays")
        result["x"] = new_x
        if "y" in data:
            y = data["y"]
            if isinstance(y, np.ndarray):
                new_y = [y]
            elif isinstance(y, tuple) and all([isinstance(yi, np.ndarray) for yi in y]):
                new_y = y
            else:
                raise ValueError("value of x should be a ndarray or a tuple of ndarrays")
            result["y"] = new_y
        return result

    def get_spec(data):
        data = check_data_type_and_to_list(data)
        feature_spec = [(feat.dtype, feat.shape[1:])
                        for feat in data["x"]]
        if "y" in data:
            label_spec = [(label.dtype, label.shape[1:])
                          for label in data["y"]]
        else:
            label_spec = None
        return (feature_spec, label_spec)

    (feature_spec, label_spec) = data_shard.rdd.map(get_spec).first()

    feature_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in feature_spec]
    label_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in label_spec] \
        if label_spec is not None else None

    assert batch_size != -1 or batch_per_thread != -1, \
        "one of batch_size and batch_per_thread should be specified"

    # todo this might be very slow
    def flatten(data):
        data = check_data_type_and_to_list(data)
        features = data["x"]

        has_label = "y" in data
        labels = data["y"] if has_label else None
        length = features[0].shape[0]

        for i in range(length):
            fs = [feat[i] for feat in features]
            if has_label:
                ls = [l[i] for l in labels]
                yield (fs, ls)
            else:
                yield (fs,)

    val_rdd = None if validation_data_shard is None \
        else validation_data_shard.rdd.flatMap(flatten)

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(flatten),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 val_rdd=val_rdd)

    return dataset


def _to_dataset(data, batch_size, batch_per_thread):
    if isinstance(data, SparkXShards):
        dataset = _xshards_to_tf_dataset(data,
                                         batch_size=batch_size,
                                         batch_per_thread=batch_per_thread)
    elif isinstance(data, Dataset):
        dataset = TFDataDataset2(data, batch_size=batch_size,
                                 batch_per_thread=batch_per_thread)
    else:
        raise ValueError("data must be a SparkXShards or an orca.data.tf.Dataset")

    return dataset


class TFOptimizerWrapper(Estimator):

    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer,
                 metrics,
                 updates, sess,
                 model_dir):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        if optimizer is not None:
            assert isinstance(optimizer, tf.train.Optimizer), \
                "optimizer is of type {}, ".format(type(self.optimizer)) + \
                "it should be an instance of tf.train.Optimizer"
            self.optimizer = ZooOptimizer(optimizer)
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

    def fit(self, data, steps,
            batch_size=32,
            validation_data=None,
            feed_dict=None,
            session_config=None):

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it not None in training"

        dataset = _to_dataset(data, batch_size=batch_size, batch_per_thread=-1)

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

        optimizer.optimize(end_trigger=MaxIteration(steps))
        return self

    def predict(self, data, batch_size=32):
        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"

        dataset = _to_dataset(data, batch_size=-1, batch_per_thread=batch_size)

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        return tfnet.predict(dataset)

    def evaluate(self, data, batch_size=32):
        assert self.metrics is not None, \
            "metrics is None, it should not be None in evaluate"

        dataset = _to_dataset(data, batch_size=-1, batch_per_thread=batch_size)

        flat_inputs = nest.flatten(self.inputs)
        flat_labels = nest.flatten(self.labels)

        return evaluate_metrics(flat_inputs + flat_labels,
                                sess=self.sess,
                                dataset=dataset, metrics=self.metrics)
