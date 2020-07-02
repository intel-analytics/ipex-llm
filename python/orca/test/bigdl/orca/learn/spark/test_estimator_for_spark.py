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
import os
import tensorflow as tf

from zoo.orca.data.tf.data import Dataset
from zoo.orca.learn.tf.estimator import Estimator
import zoo.orca.data.pandas

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


class SimpleModel(object):

    def __init__(self):
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))

        feat = tf.stack([self.user, self.item], axis=1)
        self.logits = tf.layers.dense(tf.to_float(feat), 2)

        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
                                                                          labels=self.label))


def test_estimator_graph(estimator_for_spark_fixture):
    import zoo.orca.data.pandas

    sc = estimator_for_spark_fixture

    tf.reset_default_graph()

    model = SimpleModel()
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            "y": df['label'].to_numpy()
        }
        return result

    data_shard = data_shard.transform_shard(transform)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        labels=[model.label],
        outputs=[model.logits],
        loss=model.loss,
        optimizer=tf.train.AdamOptimizer(),
        metrics={"loss": model.loss})
    est.fit(data=data_shard,
            batch_size=8,
            steps=10,
            validation_data=data_shard)

    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
        }
        return result

    data_shard = data_shard.transform_shard(transform)
    predictions = est.predict(data_shard).collect()
    print(predictions)


def test_estimator_graph_fit(estimator_for_spark_fixture):
    import zoo.orca.data.pandas
    tf.reset_default_graph()

    model = SimpleModel()
    sc = estimator_for_spark_fixture
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            "y": df['label'].to_numpy()
        }
        return result

    data_shard = data_shard.transform_shard(transform)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        labels=[model.label],
        loss=model.loss,
        optimizer=tf.train.AdamOptimizer(),
        metrics={"loss": model.loss})
    est.fit(data=data_shard,
            batch_size=8,
            steps=10,
            validation_data=data_shard)


def test_estimator_graph_evaluate(estimator_for_spark_fixture):
    import zoo.orca.data.pandas
    tf.reset_default_graph()

    model = SimpleModel()
    sc = estimator_for_spark_fixture
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            "y": df['label'].to_numpy()
        }
        return result

    data_shard = data_shard.transform_shard(transform)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        labels=[model.label],
        loss=model.loss,
        optimizer=tf.train.AdamOptimizer(),
        metrics={"loss": model.loss})
    result = est.evaluate(data_shard)
    assert "loss" in result
    print(result)


def test_estimator_graph_predict(estimator_for_spark_fixture):
    import zoo.orca.data.pandas
    tf.reset_default_graph()

    sc = estimator_for_spark_fixture

    model = SimpleModel()
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        outputs=[model.logits])

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
        }
        return result

    data_shard = data_shard.transform_shard(transform)
    predictions = est.predict(data_shard).collect()
    print(predictions)


def test_estimator_graph_fit_dataset(estimator_for_spark_fixture):
    import zoo.orca.data.pandas
    tf.reset_default_graph()
    model = SimpleModel()
    sc = estimator_for_spark_fixture
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            "y": df['label'].to_numpy()
        }
        return result

    data_shard = data_shard.transform_shard(transform)
    dataset = Dataset.from_tensor_slices(data_shard)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        labels=[model.label],
        loss=model.loss,
        optimizer=tf.train.AdamOptimizer(),
        metrics={"loss": model.loss})
    est.fit(data=dataset,
            batch_size=8,
            steps=10,
            validation_data=dataset)

    result = est.evaluate(dataset, batch_size=4)
    assert 'loss' in result


def test_estimator_graph_predict_dataset(estimator_for_spark_fixture):

    sc = estimator_for_spark_fixture
    tf.reset_default_graph()

    model = SimpleModel()
    file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
    data_shard = zoo.orca.data.pandas.read_csv(file_path, sc)

    est = Estimator.from_graph(
        inputs=[model.user, model.item],
        outputs=[model.logits])

    def transform(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
        }
        return result

    data_shard = data_shard.transform_shard(transform)
    dataset = Dataset.from_tensor_slices(data_shard)
    predictions = est.predict(dataset).collect()
    assert len(predictions) == 10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
