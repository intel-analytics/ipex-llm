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

import shutil
from unittest import TestCase

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.feature.common import *
from zoo.orca.learn.bigdl import Estimator
from bigdl.optim.optimizer import *
from zoo.pipeline.api.keras import layers as ZLayer
from zoo.pipeline.api.keras.models import Model as ZModel
from zoo.orca.data import SparkXShards
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch
from zoo.orca.data.pandas import read_csv


class TestEstimatorForKeras(TestCase):
    def get_estimator_df(self):
        self.sc = init_nncontext()
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        self.sqlContext = SQLContext(self.sc)
        df = self.sqlContext.createDataFrame(data, schema)
        return df

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
        from zoo.pipeline.nnframes import NNModel
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df = self.get_estimator_df()
        est = Estimator.from_bigdl(model=linear_model, loss=mse_criterion, optimizer=Adam(),
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        res0 = est.predict(df)
        res0_c = res0.collect()
        est.fit(df, 1, batch_size=4)
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
            "x": [np.expand_dims(np.array(feature_label_tuple[0][0]), axis=0),
                  np.expand_dims(np.array(feature_label_tuple[0][1]), axis=0)],
            "y": [np.expand_dims(np.array(feature_label_tuple[1][0]), axis=0),
                  np.expand_dims(np.array(feature_label_tuple[1][1]), axis=0)]
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

    def test_nnEstimator_multiInput(self):
        zx1 = ZLayer.Input(shape=(1,))
        zx2 = ZLayer.Input(shape=(1,))
        zz = ZLayer.merge([zx1, zx2], mode="concat")
        zy = ZLayer.Dense(2)(zz)
        zmodel = ZModel([zx1, zx2], zy)

        criterion = MSECriterion()
        df = self.get_estimator_df()
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
        est.fit(df, 1, batch_size=4, feature_cols=["user", "age"], labels_cols=["label1", "label2"])
        result = est.predict(df, feature_cols=["user", "age"])
        result_c = result.collect()
        assert type(result).__name__ == 'DataFrame'

    def test_xshards_spark_estimator(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")

        def transform(df):
            result = {
                "x": [df['user'].to_numpy(), df['item'].to_numpy()],
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
                                             loss=ClassNLLCriterion(), model_dir=temp_dir_name,
                                             feature_preprocessing=SeqToTensor([2]),
                                             label_preprocessing=SeqToTensor([1]))
            estimator.set_constant_gradient_clipping(0.1, 1.2)
            r1 = estimator.predict(data=data_shard)
            r_c = r1.collect()
            estimator.set_tensorboard(log_dir=temp_dir_name, app_name="test")
            estimator.fit(data=data_shard, epochs=5, batch_size=8, val_data=data_shard,
                          val_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
            summary = estimator.get_train_summary(tag="Loss")
            temp_path = os.path.join(temp_dir_name, "save_model")
            estimator.save(temp_path)
            estimator.evaluate(data=data_shard, validation_methods=[Accuracy()], batch_size=8)
            result = estimator.predict(data=data_shard)
            assert type(result).__name__ == 'SparkXShards'
            result_c = result.collect()
            df = self.get_estimator_df2()
            r0 = estimator.predict(df)
            r0_c = r0.collect()
            assert type(r0).__name__ == 'DataFrame'
            for idx in range(len(r0_c)):
                assert abs(r0_c[idx]["prediction"][0] - result_c[0]["prediction"][idx][0]) == 0
                assert abs(r0_c[idx]["prediction"][1] - result_c[0]["prediction"][idx][1]) == 0
            estimator.fit(data=df, epochs=6, batch_size=8, val_data=df, val_methods=[Accuracy()],
                          val_trigger=EveryEpoch())
            summary = estimator.get_train_summary()

            # test load from checkpoint
            est2 = Estimator.from_bigdl(model=Sequential(), optimizer=None, loss=None,
                                        model_dir=None)
            est2.load(temp_dir_name, loss=ClassNLLCriterion(), is_checkpoint=True)
            r2 = est2.predict(data=data_shard)
            r2_c = r2.collect()
            assert (result_c[0]["prediction"] == r2_c[0]["prediction"]).all()
            # resume training
            est2.fit(data=data_shard, epochs=10, batch_size=8, val_data=data_shard,
                     val_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
            est2.evaluate(data=data_shard, validation_methods=[Accuracy()], batch_size=8)
            # test load from saved model
            est3 = Estimator.from_bigdl(model=Sequential(), optimizer=None, loss=None,
                                        model_dir=None)
            est3.load(temp_path, optimizer=optim_method, loss=ClassNLLCriterion())
            r3 = est3.predict(data=data_shard)
            r3_c = r3.collect()
            assert (r3_c[0]["prediction"] == r2_c[0]["prediction"]).all()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
