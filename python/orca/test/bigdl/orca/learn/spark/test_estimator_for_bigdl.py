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

from bigdl.dllib.nn.criterion import *
from bigdl.dllib.nn.layer import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf

from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.common import *
from bigdl.orca.learn.bigdl import Estimator
from bigdl.dllib.optim.optimizer import Adam, SGD, ValidationSummary
from bigdl.dllib.keras import layers as ZLayer
from bigdl.dllib.keras.models import Model as ZModel
from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.metrics import Accuracy, MAE
from bigdl.orca.learn.trigger import EveryEpoch, SeveralIteration
from bigdl.orca.data.pandas import read_csv


class TestEstimatorForKeras(TestCase):
    def get_estimator_df(self):
        self.sc = init_nncontext()
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])
        val_data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])
        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        self.sqlContext = SQLContext(self.sc)
        df = self.sqlContext.createDataFrame(data, schema)
        val_df = self.sqlContext.createDataFrame(val_data, schema)
        return df, val_df

    def get_estimator_df2(self):
        self.sc = init_nncontext()
        data = self.sc.parallelize([
            ((0.0, 0.0), 1.0),
            ((1.0, 1.0), 2.0),
            ((2.0, 2.0), 1.0),
            ((3.0, 3.0), 2.0),
            ((4.0, 4.0), 1.0),
            ((5.0, 5.0), 2.0),
            ((6.0, 6.0), 1.0),
            ((7.0, 7.0), 2.0),
            ((8.0, 8.0), 1.0),
            ((9.0, 9.0), 2.0)
        ])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        self.sqlContext = SQLContext(self.sc)
        df = self.sqlContext.createDataFrame(data, schema)
        return df

    def test_nnEstimator(self):
        from bigdl.dllib.nnframes import NNModel
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df, _ = self.get_estimator_df()
        est = Estimator.from_bigdl(model=linear_model, loss=mse_criterion, optimizer=Adam(),
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        res0 = est.predict(df)
        res0_c = res0.collect()
        est.fit(df, 2, batch_size=4)
        nn_model = NNModel(est.get_model(), feature_preprocessing=SeqToTensor([2]))
        res1 = nn_model.transform(df)
        res2 = est.predict(df)
        res1_c = res1.collect()
        res2_c = res2.collect()
        assert type(res1).__name__ == 'DataFrame'
        assert type(res2).__name__ == 'DataFrame'
        assert len(res1_c) == len(res2_c)
        for idx in range(len(res1_c)):
            assert res1_c[idx]["prediction"] == res2_c[idx]["prediction"]
        with tempfile.TemporaryDirectory() as tempdirname:
            temp_path = os.path.join(tempdirname, "model")
            est.save(temp_path)
            est2 = Estimator.from_bigdl(model=linear_model, loss=mse_criterion)
            est2.load(temp_path, optimizer=Adam(), loss=mse_criterion,
                      feature_preprocessing=SeqToTensor([2]), label_preprocessing=SeqToTensor([2]))
            est2.set_constant_gradient_clipping(0.1, 1.2)
            est2.clear_gradient_clipping()
            res3 = est2.predict(df)
            res3_c = res3.collect()
            assert type(res3).__name__ == 'DataFrame'
            assert len(res1_c) == len(res3_c)
            for idx in range(len(res1_c)):
                assert res1_c[idx]["prediction"] == res3_c[idx]["prediction"]
            est2.fit(df, 4, batch_size=4)

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])
        data_shard = SparkXShards(data)
        data_shard = data_shard.transform_shard(lambda feature_label_tuple: {
            "x": np.stack([np.expand_dims(np.array(feature_label_tuple[0][0]), axis=0),
                           np.expand_dims(np.array(feature_label_tuple[0][1]), axis=0)], axis=1),
            "y": np.stack([np.expand_dims(np.array(feature_label_tuple[1][0]), axis=0),
                           np.expand_dims(np.array(feature_label_tuple[1][1]), axis=0)], axis=1)
        })
        res4 = est.predict(data_shard)
        res4_c = res4.collect()
        assert type(res4).__name__ == 'SparkXShards'
        for idx in range(len(res4_c)):
            assert abs(res4_c[idx]["prediction"][0][0] - res3_c[idx]["prediction"][0]) == 0
            assert abs(res4_c[idx]["prediction"][0][1] - res3_c[idx]["prediction"][1]) == 0
        est.fit(data_shard, 1, batch_size=4)
        res5 = est.predict(data_shard)
        res5_c = res5.collect()
        res6 = est.predict(df)
        res6_c = res6.collect()
        for idx in range(len(res5_c)):
            assert abs(res5_c[idx]["prediction"][0][0] - res6_c[idx]["prediction"][0]) == 0
            assert abs(res5_c[idx]["prediction"][0][1] - res6_c[idx]["prediction"][1]) == 0

    def test_nnEstimator_evaluation(self):
        df = self.get_estimator_df2()
        linear_model = Sequential().add(Linear(2, 2)).add(LogSoftMax())

        est = Estimator.from_bigdl(model=linear_model, loss=ClassNLLCriterion(), optimizer=Adam(),
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([1]),
                                   metrics=Accuracy())
        est.fit(data=df, epochs=10, batch_size=8)
        result = est.evaluate(df, batch_size=8)

        shift = udf(lambda p: float(p.index(max(p))), DoubleType())
        pred = est.predict(df).withColumn("prediction", shift(col('prediction'))).cache()

        correct = pred.filter("label=prediction").count()
        overall = pred.count()
        accuracy = correct * 1.0 / overall
        assert accuracy == round(result['Top1Accuracy'], 2)

    def test_nnEstimator_multiInput(self):
        zx1 = ZLayer.Input(shape=(1,))
        zx2 = ZLayer.Input(shape=(1,))
        zz = ZLayer.merge([zx1, zx2], mode="concat")
        zy = ZLayer.Dense(2)(zz)
        zmodel = ZModel([zx1, zx2], zy)

        criterion = MSECriterion()
        df, _ = self.get_estimator_df()
        estimator = Estimator.from_bigdl(model=zmodel, loss=criterion,
                                         feature_preprocessing=[[1], [1]])
        estimator.fit(df, epochs=5, batch_size=4)
        pred = estimator.predict(df)
        pred_data = pred.collect()
        assert type(pred).__name__ == 'DataFrame'

    def test_nnEstimator_multiInput_cols(self):
        from pyspark.ml.linalg import Vectors
        from pyspark.sql import SparkSession

        spark = SparkSession \
            .builder \
            .getOrCreate()

        df = spark.createDataFrame(
            [(1, 35, 109.0, Vectors.dense([2.0, 5.0, 0.5, 0.5]), 1.0),
             (2, 58, 2998.0, Vectors.dense([4.0, 10.0, 0.5, 0.5]), 2.0),
             (3, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0),
             (4, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0)],
            ["user", "age", "income", "history", "label"])

        x1 = ZLayer.Input(shape=(1,))
        x2 = ZLayer.Input(shape=(2,))
        x3 = ZLayer.Input(shape=(2, 2,))

        user_embedding = ZLayer.Embedding(5, 10)(x1)
        flatten = ZLayer.Flatten()(user_embedding)
        dense1 = ZLayer.Dense(2)(x2)
        gru = ZLayer.LSTM(4, input_shape=(2, 2))(x3)

        merged = ZLayer.merge([flatten, dense1, gru], mode="concat")
        zy = ZLayer.Dense(2)(merged)

        zmodel = ZModel([x1, x2, x3], zy)
        criterion = ClassNLLCriterion()
        est = Estimator.from_bigdl(model=zmodel, loss=criterion, optimizer=Adam(learningrate=0.1),
                                   feature_preprocessing=[[1], [2], [2, 2]])
        est.fit(df, epochs=1, batch_size=4, feature_cols=["user", "age", "income", "history"])

        res = est.predict(df, feature_cols=["user", "age", "income", "history"])
        res_c = res.collect()
        assert type(res).__name__ == 'DataFrame'

    def test_nnEstimator_multiOutput_cols(self):
        from pyspark.ml.linalg import Vectors
        from pyspark.sql import SparkSession

        spark = SparkSession \
            .builder \
            .getOrCreate()

        df = spark.createDataFrame(
            [(1.0, 2.0, 1.0, 2.0),
             (2.0, 2.0, 2.0, 1.0),
             (3.0, 2.0, 1.0, 2.0),
             (4.0, 1.0, 1.0, 2.0)],
            ["user", "age", "label1", "label2"])
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        est = Estimator.from_bigdl(model=linear_model, loss=mse_criterion, optimizer=Adam(),
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        est.fit(df, 1, batch_size=4, feature_cols=["user", "age"], label_cols=["label1", "label2"])
        result = est.predict(df, feature_cols=["user", "age"])
        result_c = result.collect()
        assert type(result).__name__ == 'DataFrame'

    def test_nnEstimator_fit_with_train_val_summary(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        df, val_df = self.get_estimator_df()
        from bigdl.orca.learn.metrics import MAE
        est = Estimator.from_bigdl(model=model, loss=criterion, optimizer=Adam(), metrics=[MAE()],
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        tmp_dir = tempfile.mkdtemp()
        est.set_tensorboard(log_dir=tmp_dir, app_name="estTest")

        est.fit(df, epochs=5, batch_size=4, validation_data=val_df, validation_trigger=EveryEpoch(),
                checkpoint_trigger=SeveralIteration(1))

        res = est.predict(df)
        loss_result = est.get_train_summary("Loss")
        mae_result = est.get_validation_summary("MAE")
        assert type(res).__name__ == 'DataFrame'
        assert len(loss_result) == 5
        assert len(mae_result) == 4

    def test_xshards_spark_estimator(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

        def transform(df):
            result = {
                "x": np.stack([df['user'].to_numpy(), df['item'].to_numpy()], axis=1),
                "y": df['label'].to_numpy()
            }
            return result

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)
        model = Sequential()
        model.add(Linear(2, 2))
        model.add(LogSoftMax())
        optim_method = SGD(learningrate=0.01)
        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_bigdl(model=model, optimizer=optim_method,
                                             loss=ClassNLLCriterion(),
                                             metrics=Accuracy(),
                                             model_dir=temp_dir_name,
                                             feature_preprocessing=SeqToTensor([2]),
                                             label_preprocessing=SeqToTensor([1]))
            estimator.set_constant_gradient_clipping(0.1, 1.2)
            r1 = estimator.predict(data=data_shard)
            r_c = r1.collect()
            estimator.set_tensorboard(log_dir=temp_dir_name, app_name="test")
            estimator.fit(data=data_shard, epochs=5, batch_size=8, validation_data=data_shard,
                          checkpoint_trigger=EveryEpoch())
            summary = estimator.get_train_summary(tag="Loss")
            temp_path = os.path.join(temp_dir_name, "save_model")
            estimator.save(temp_path)
            with self.assertRaises(Exception) as context:
                Estimator.from_bigdl(model=model, optimizer=optim_method,
                                     loss=ClassNLLCriterion(),
                                     metrics=['accuracy'],
                                     model_dir=temp_dir_name,
                                     feature_preprocessing=SeqToTensor([2]),
                                     label_preprocessing=SeqToTensor([1]))
            self.assertTrue('Only orca metrics are supported, but get str' in
                            str(context.exception))
            eval_result = estimator.evaluate(data=data_shard,
                                             batch_size=8)
            assert isinstance(eval_result, dict)
            result = estimator.predict(data=data_shard)
            assert type(result).__name__ == 'SparkXShards'
            result_c = result.collect()
            df = self.get_estimator_df2()
            r0 = estimator.predict(df)
            r0_c = r0.collect()
            assert type(r0).__name__ == 'DataFrame'
            for idx in range(len(r0_c)):
                assert abs(r0_c[idx]["prediction"][0] - result_c[0]["prediction"][idx][0]) <= 1e-06
                assert abs(r0_c[idx]["prediction"][1] - result_c[0]["prediction"][idx][1]) <= 1e-06
            estimator.fit(data=df, epochs=6, batch_size=8, validation_data=df,
                          validation_trigger=EveryEpoch())
            summary = estimator.get_train_summary("Loss")

            # test load from checkpoint
            est2 = Estimator.from_bigdl(model=Sequential(), optimizer=None, loss=None,
                                        metrics=[Accuracy()],
                                        model_dir=None)
            est2.load(temp_dir_name, loss=ClassNLLCriterion(), is_checkpoint=True)
            r2 = est2.predict(data=data_shard)
            r2_c = r2.collect()
            assert (result_c[0]["prediction"] == r2_c[0]["prediction"]).all()
            # resume training
            est2.fit(data=data_shard, epochs=10, batch_size=8, validation_data=data_shard,
                     checkpoint_trigger=EveryEpoch())
            est2.evaluate(data=data_shard, batch_size=8)
            # test load from saved model
            est3 = Estimator.from_bigdl(model=Sequential(), optimizer=None, loss=None,
                                        model_dir=None)
            est3.load(temp_path, optimizer=optim_method, loss=ClassNLLCriterion())
            r3 = est3.predict(data=data_shard)
            r3_c = r3.collect()
            assert (r3_c[0]["prediction"] == r2_c[0]["prediction"]).all()

    def test_xshards_spark_estimator_multi_inputs(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

        def transform(df):
            result = {
                "x": [np.expand_dims(df['user'].to_numpy(), axis=1),
                      np.expand_dims(df['item'].to_numpy(), axis=1)],
                "y": df['label'].to_numpy()
            }
            return result

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)
        zx1 = ZLayer.Input(shape=(1,))
        zx2 = ZLayer.Input(shape=(1,))
        zz = ZLayer.merge([zx1, zx2], mode="concat")
        zy = ZLayer.Dense(2)(zz)
        model = ZModel([zx1, zx2], zy)

        optim_method = SGD(learningrate=0.01)
        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_bigdl(model=model, optimizer=optim_method,
                                             loss=ClassNLLCriterion(),
                                             metrics=[Accuracy()],
                                             model_dir=temp_dir_name)
            estimator.set_constant_gradient_clipping(0.1, 1.2)
            r1 = estimator.predict(data=data_shard)
            r_c = r1.collect()
            estimator.set_tensorboard(log_dir=temp_dir_name, app_name="test")
            estimator.fit(data=data_shard, epochs=5, batch_size=8, validation_data=data_shard,
                          checkpoint_trigger=EveryEpoch())
            summary = estimator.get_train_summary(tag="Loss")
            temp_path = os.path.join(temp_dir_name, "save_model")
            estimator.save(temp_path)
            eval_result = estimator.evaluate(data=data_shard,
                                             batch_size=8)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
