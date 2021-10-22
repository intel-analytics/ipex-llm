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

import shutil
from unittest import TestCase

import tensorflow as tf

from bigdl.orca import OrcaContext
from bigdl.orca.learn.trigger import SeveralIteration
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.dllib.nncontext import *
from bigdl.orca.learn.utils import convert_predict_rdd_to_dataframe


class TestEstimatorForKeras(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        OrcaContext.train_data_store = "DRAM"

    def create_model(self):
        user = tf.keras.layers.Input(shape=[1])
        item = tf.keras.layers.Input(shape=[1])

        feat = tf.keras.layers.concatenate([user, item], axis=1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)

        model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_model_lr_schedule(self, init_lr, decay_steps, decay_rate):
        x = tf.keras.layers.Input(shape=[8])

        predictions = tf.keras.layers.Dense(1, use_bias=False,
                                            kernel_initializer=tf.ones_initializer())(x)

        model = tf.keras.models.Model(inputs=[x], outputs=predictions)
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, decay_steps, decay_rate)
        optimizer = tf.keras.optimizers.SGD(schedule)
        model.compile(optimizer=optimizer,
                      loss=lambda label, pred: tf.reduce_mean(pred - label))
        return model

    def create_model_with_clip(self):
        user = tf.keras.layers.Input(shape=[1])
        item = tf.keras.layers.Input(shape=[1])

        feat = tf.keras.layers.concatenate([user, item], axis=1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)

        model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop",
            clipnorm=1.2,
            clipvalue=0.2
        )
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def test_estimator_keras_xshards(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        assert predictions[0]['prediction'].shape[1] == 2

    def test_estimator_keras_xshards_options(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        # train with no validation
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10)
        # train with different optimizer
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10
                )
        # train with session config
        tf_session_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                                           intra_op_parallelism_threads=1)
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                session_config=tf_session_config
                )
        # train with model dir
        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, "model")
        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)
        assert len(os.listdir(model_dir)) > 0
        shutil.rmtree(temp)

    def test_estimator_keras_xshards_clip(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model_with_clip()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

    def test_estimator_keras_xshards_checkpoint(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, "test_model")

        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=6,
                validation_data=data_shard,
                checkpoint_trigger=SeveralIteration(4))

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        tf.reset_default_graph()

        model = self.create_model()

        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
        est.load_orca_checkpoint(model_dir)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard,
                checkpoint_trigger=SeveralIteration(4))

        eval_result = est.evaluate(data_shard)
        print(eval_result)
        shutil.rmtree(temp)

    def test_estimator_keras_dataframe(self):
        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=df,
                batch_size=8,
                epochs=4,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                validation_data=df)

        eval_result = est.evaluate(df, feature_cols=['user', 'item'], label_cols=['label'])
        assert 'acc Top1Accuracy' in eval_result

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 48

    def test_estimator_keras_dataframe_no_fit(self):
        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)

        eval_result = est.evaluate(df, feature_cols=['user', 'item'], label_cols=['label'])
        assert 'acc Top1Accuracy' in eval_result

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 48

    def test_estimator_keras_tf_dataset(self):
        tf.reset_default_graph()

        model = self.create_model()

        dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100, 1)),
                                                      np.random.randint(0, 50, size=(100, 1)),
                                                      np.ones(shape=(100,), dtype=np.int32)))
        dataset = dataset.map(lambda user, item, label: [(user, item), label])
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=dataset,
                batch_size=8,
                epochs=10,
                validation_data=dataset)

        eval_result = est.evaluate(dataset)
        assert 'acc Top1Accuracy' in eval_result

    def test_estimator_keras_tensorboard(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, "test_model")

        est = Estimator.from_keras(keras_model=model, model_dir=model_dir)

        assert est.get_train_summary("Loss") is None
        assert est.get_validation_summary("Top1Accuracy") is None

        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        train_loss = est.get_train_summary("Loss")
        assert len(train_loss) > 0
        val_scores = est.get_validation_summary("Top1Accuracy")
        assert len(val_scores) > 0

        tf.reset_default_graph()
        # no model dir
        model = self.create_model()
        est = Estimator.from_keras(keras_model=model)
        log_dir = os.path.join(temp, "log")
        est.set_tensorboard(log_dir, "test")

        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        assert os.path.exists(os.path.join(log_dir, "test/train"))
        assert os.path.exists(os.path.join(log_dir, "test/validation"))

        train_loss = est.get_train_summary("Loss")
        val_scores = est.get_validation_summary("Loss")
        assert len(train_loss) > 0
        assert len(val_scores) > 0
        shutil.rmtree(temp)

    def test_convert_predict_list_of_array(self):
        tf.reset_default_graph()

        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        rdd = sc.parallelize([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        df = rdd.toDF(["feature", "label", "c"])
        predict_rdd = df.rdd.map(lambda row: [np.array([1, 2]), np.array(0)])
        resultDF = convert_predict_rdd_to_dataframe(df, predict_rdd)
        resultDF.printSchema()
        print(resultDF.collect()[0])
        predict_rdd = df.rdd.map(lambda row: np.array(1))
        resultDF = convert_predict_rdd_to_dataframe(df, predict_rdd)
        resultDF.printSchema()
        print(resultDF.collect()[0])

    def test_estimator_keras_save_load(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        temp = tempfile.mkdtemp()
        model_path = os.path.join(temp, 'test.h5')
        est.save_keras_model(model_path)

        tf.reset_default_graph()

        est = Estimator.load_keras_model(model_path)

        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        assert predictions[0]['prediction'].shape[1] == 2
        shutil.rmtree(temp)

    def test_estimator_keras_weights_save_load(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=data_shard,
                batch_size=8,
                epochs=10,
                validation_data=data_shard)

        eval_result = est.evaluate(data_shard)
        print(eval_result)

        temp = tempfile.mkdtemp()
        model_path = os.path.join(temp, 'test.h5')
        est.save_keras_weights(model_path)

        tf.reset_default_graph()

        model = self.create_model()

        est = Estimator.from_keras(model)
        est.load_keras_weights(model_path)

        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        assert predictions[0]['prediction'].shape[1] == 2
        shutil.rmtree(temp)

    def test_estimator_keras_learning_rate_schedule(self):
        tf.reset_default_graph()

        # loss = reduce_sum(w)
        # dloss/dw = 1
        model = self.create_model_lr_schedule(0.1, 1, 0.1)

        dataset = tf.data.Dataset.from_tensor_slices((np.ones((16, 8)),
                                                      np.zeros((16, 1))))
        est = Estimator.from_keras(keras_model=model)
        weights_before = model.get_weights()[0]
        est.fit(data=dataset,
                batch_size=8,
                epochs=1,
                validation_data=dataset)

        sess = tf.keras.backend.get_session()
        iteartion = sess.run(model.optimizer.iterations)
        weights_after = model.get_weights()[0]
        first_step = weights_before - 0.1
        second_step = first_step - 0.01

        assert iteartion == 2
        assert np.allclose(second_step, weights_after)

    def test_estimator_keras_with_bigdl_optim_method(self):
        tf.reset_default_graph()

        model = self.create_model()

        dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100, 1)),
                                                      np.random.randint(0, 50, size=(100, 1)),
                                                      np.ones(shape=(100,), dtype=np.int32)))
        dataset = dataset.map(lambda user, item, label: [(user, item), label])
        from bigdl.orca.learn.optimizers import SGD
        from bigdl.orca.learn.optimizers.schedule import Plateau
        sgd = SGD(learningrate=0.1,
                  learningrate_schedule=Plateau("score",
                                                factor=0.1,
                                                patience=10,
                                                mode="min", ))
        est = Estimator.from_keras(keras_model=model, optimizer=sgd)
        est.fit(data=dataset,
                batch_size=8,
                epochs=10,
                validation_data=dataset)

    def test_submodel_in_keras_squential(self):
        mnet = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                 include_top=False,
                                                 weights='imagenet')

        model = tf.keras.Sequential([
            mnet,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        dataset = tf.data.Dataset.from_tensor_slices((np.random.randn(16, 160, 160, 3),
                                                      np.random.randint(0, 1000, (16, 1))))
        est = Estimator.from_keras(keras_model=model)
        est.fit(data=dataset,
                batch_size=4,
                epochs=1,
                validation_data=dataset)

    def test_estimator_keras_xshards_with_mem_type(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)

        est = Estimator.from_keras(keras_model=model)
        OrcaContext.train_data_store = "DISK_2"
        est.fit(data=data_shard,
                batch_size=4,
                epochs=10,
                validation_data=data_shard
                )

        eval_result = est.evaluate(data_shard)
        print(eval_result)
        OrcaContext.train_data_store = "DRAM"

    def test_estimator_keras_xshards_disk_featureset_trigger(self):
        import bigdl.orca.data.pandas

        tf.reset_default_graph()

        model = self.create_model()
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            result = {
                "x": (df['user'].to_numpy().reshape([-1, 1]),
                      df['item'].to_numpy().reshape([-1, 1])),
                "y": df['label'].to_numpy()
            }
            return result

        data_shard = data_shard.transform_shard(transform)
        from bigdl.dllib.optim.optimizer import SeveralIteration
        from bigdl.dllib.utils.triggers import SeveralIteration as ZSeveralIteration
        from bigdl.dllib.utils.triggers import MinLoss as ZMinLoss
        from bigdl.dllib.utils.triggers import TriggerAnd as ZTriggerAnd
        est = Estimator.from_keras(keras_model=model)
        OrcaContext.train_data_store = "DISK_2"
        with self.assertRaises(Exception) as context:
            est.fit(data=data_shard,
                    batch_size=4,
                    epochs=10,
                    validation_data=data_shard,
                    checkpoint_trigger=SeveralIteration(2))
        self.assertTrue('Please use a trigger defined in bigdl.dllib.utils.triggers'
                        in str(context.exception))

        est.fit(data=data_shard,
                batch_size=4,
                epochs=10,
                validation_data=data_shard,
                checkpoint_trigger=ZTriggerAnd(ZSeveralIteration(2), ZMinLoss(0.2)))
        OrcaContext.train_data_store = "DRAM"

    def test_estimator_keras_dataframe_mem_type(self):
        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)
        OrcaContext.train_data_store = "DISK_2"
        est.fit(data=df,
                batch_size=4,
                epochs=4,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                validation_data=df)

        eval_result = est.evaluate(df, feature_cols=['user', 'item'], label_cols=['label'])
        assert 'acc Top1Accuracy' in eval_result

        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 48
        OrcaContext.train_data_store = "DRAM"

    def test_estimator_keras_get_model(self):
        tf.reset_default_graph()

        model = self.create_model()
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        from pyspark.sql.functions import array
        df = df.withColumn('user', array('user')) \
            .withColumn('item', array('item'))

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=df,
                batch_size=4,
                epochs=4,
                feature_cols=['user', 'item'],
                label_cols=['label'],
                validation_data=df)
        assert est.get_model() is model


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
