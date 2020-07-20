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
from pyspark.sql.context import SQLContext

from unittest import TestCase

from zoo import init_nncontext
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


class TestEstimatorForGraph(TestCase):

    def test_estimator_graph(self):
        import zoo.orca.data.pandas

        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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
                epochs=10,
                validation_data=data_shard)

        data_shard = zoo.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        print(predictions)

    def test_estimator_graph_fit(self):
        import zoo.orca.data.pandas
        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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
                epochs=10,
                validation_data=data_shard)

    def test_estimator_graph_evaluate(self):
        import zoo.orca.data.pandas
        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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

    def test_estimator_graph_predict(self):
        import zoo.orca.data.pandas
        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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

    def test_estimator_graph_fit_clip(self):
        import zoo.orca.data.pandas
        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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
            clip_norm=1.2,
            metrics={"loss": model.loss})
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        est = Estimator.from_graph(
            inputs=[model.user, model.item],
            labels=[model.label],
            loss=model.loss,
            optimizer=tf.train.AdamOptimizer(),
            clip_value=0.2,
            metrics={"loss": model.loss})
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

    def test_estimator_graph_fit_dataset(self):
        import zoo.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()

        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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
                epochs=10,
                validation_data=dataset)

        result = est.evaluate(dataset, batch_size=4)
        assert 'loss' in result

    def test_estimator_graph_predict_dataset(self):
        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)

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

    def test_estimator_graph_dataframe(self):

        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)

        est = Estimator.from_graph(
            inputs=[model.user, model.item],
            labels=[model.label],
            outputs=[model.logits],
            loss=model.loss,
            optimizer=tf.train.AdamOptimizer(),
            metrics={"loss": model.loss})

        est.fit(data=df,
                batch_size=8,
                epochs=10,
                feature_cols=['user', 'item'],
                labels_cols=['label'],
                validation_data=df)

        result = est.evaluate(df, batch_size=4, feature_cols=['user', 'item'],
                              labels_cols=['label'])
        print(result)

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 10

    def test_estimator_graph_dataframe_exception(self):

        tf.reset_default_graph()

        model = SimpleModel()
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)

        est = Estimator.from_graph(
            inputs=[model.user, model.item],
            labels=[model.label],
            outputs=[model.logits],
            loss=model.loss,
            optimizer=tf.train.AdamOptimizer(),
            metrics={"loss": model.loss})

        with self.assertRaises(Exception) as context:
            est.fit(data=df,
                    batch_size=8,
                    epochs=10,
                    feature_cols=['user', 'item'],
                    validation_data=df)
        self.assertTrue('label columns is None; it should not be None in training'
                        in str(context.exception))

        est.fit(data=df,
                batch_size=8,
                epochs=10,
                feature_cols=['user', 'item'],
                labels_cols=['label']
                )
        with self.assertRaises(Exception) as context:
            predictions = est.predict(df, batch_size=4).collect()
        self.assertTrue('feature columns is None; it should not be None in prediction'
                        in str(context.exception))

        with self.assertRaises(Exception) as context:
            est.fit(data=df,
                    batch_size=8,
                    epochs=10,
                    feature_cols=['user', 'item'],
                    labels_cols=['label'],
                    validation_data=[1, 2, 3])
        self.assertTrue('train data and validation data should be both Spark DataFrame'
                        in str(context.exception))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
