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

import pytest
import shutil
import errno
import string
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.keras.optimizers import Adam as KAdam
from zoo.pipeline.api.keras import layers as ZLayer
from zoo.pipeline.api.keras.models import Model as ZModel
from zoo.feature.common import *
from zoo.feature.image import *
from zoo.util.tf import *


class TestNNClassifer():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
        self.sc = init_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        assert(self.sc.appName == "testNNClassifer")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def get_estimator_df(self):
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        return df

    def get_classifier_df(self):
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        return df

    def test_nnEstimator_construct_with_differnt_params(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df = self.get_estimator_df()
        for e in [NNEstimator(linear_model, mse_criterion),
                  NNEstimator(linear_model, mse_criterion, [2], [2]),
                  NNEstimator(linear_model, mse_criterion, SeqToTensor([2]), SeqToTensor([2]))]:
            nnModel = e.setBatchSize(4).setMaxEpoch(1).fit(df)
            res = nnModel.transform(df)
            assert type(res).__name__ == 'DataFrame'

    def test_nnClassifier_construct_with_differnt_params(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df = self.get_classifier_df()
        for e in [NNClassifier(linear_model, mse_criterion),
                  NNClassifier(linear_model, mse_criterion, [2]),
                  NNClassifier(linear_model, mse_criterion, SeqToTensor([2]))]:
            nnModel = e.setBatchSize(4).setMaxEpoch(1).fit(df)
            res = nnModel.transform(df)
            assert type(res).__name__ == 'DataFrame'

    def test_nnModel_construct_with_differnt_params(self):
        linear_model = Sequential().add(Linear(2, 2))
        df = self.get_estimator_df()
        for e in [NNModel(linear_model),
                  NNModel(linear_model, [2]),
                  NNModel(linear_model, SeqToTensor([2]))]:
            res = e.transform(df)
            assert type(res).__name__ == 'DataFrame'
            assert e.getBatchSize() == 4

    def test_nnClassiferModel_construct_with_differnt_params(self):
        linear_model = Sequential().add(Linear(2, 2))
        df = self.get_classifier_df()
        for e in [NNClassifierModel(linear_model),
                  NNClassifierModel(linear_model, [2]),
                  NNClassifierModel(linear_model, SeqToTensor([2]))]:
            res = e.transform(df)
            assert type(res).__name__ == 'DataFrame'
            assert e.getBatchSize() == 4

    def test_all_set_get_methods(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()

        estimator = NNEstimator(linear_model, mse_criterion, SeqToTensor([2]), SeqToTensor([2]))
        assert estimator.setBatchSize(30).getBatchSize() == 30
        assert estimator.setMaxEpoch(40).getMaxEpoch() == 40
        assert estimator.setLearningRate(1e-4).getLearningRate() == 1e-4
        assert estimator.setFeaturesCol("abcd").getFeaturesCol() == "abcd"
        assert estimator.setLabelCol("xyz").getLabelCol() == "xyz"
        assert isinstance(estimator.setOptimMethod(Adam()).getOptimMethod(), Adam)

        nn_model = NNModel(linear_model, SeqToTensor([2]))
        assert nn_model.setBatchSize(20).getBatchSize() == 20

        linear_model = Sequential().add(Linear(2, 2))
        classNLL_criterion = ClassNLLCriterion()
        classifier = NNClassifier(linear_model, classNLL_criterion, SeqToTensor([2]))
        assert classifier.setBatchSize(20).getBatchSize() == 20
        assert classifier.setMaxEpoch(50).getMaxEpoch() == 50
        assert classifier.setLearningRate(1e-5).getLearningRate() == 1e-5
        assert classifier.setLearningRateDecay(1e-9).getLearningRateDecay() == 1e-9
        assert classifier.setCachingSample(False).isCachingSample() is False

        nn_classifier_model = NNClassifierModel(linear_model, SeqToTensor([2]))
        assert nn_classifier_model.setBatchSize((20)).getBatchSize() == 20

    def test_nnEstimator_fit_nnmodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), ArrayToTensor([2]))\
            .setBatchSize(4).setLearningRate(0.2).setMaxEpoch(40)

        df = self.get_estimator_df()
        nnModel = estimator.fit(df)
        assert nnModel.getBatchSize() == 4

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("nnModelDF")  # Compatible with spark 1.6
        results = self.sqlContext.table("nnModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert_allclose(row_label[0], row_prediction[0], atol=0, rtol=1e-1)
            assert_allclose(row_label[1], row_prediction[1], atol=0, rtol=1e-1)

    def test_nnEstimator_fit_gradient_clipping(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), ArrayToTensor([2])) \
            .setBatchSize(4).setLearningRate(0.2).setMaxEpoch(2)\
            .setConstantGradientClipping(0.1, 0.2)

        df = self.get_estimator_df()
        estimator.fit(df)
        estimator.clearGradientClipping()
        estimator.fit(df)
        estimator.setGradientClippingByL2Norm(1.2)
        estimator.fit(df)

    def test_nnEstimator_fit_with_Cache_Disk(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), ArrayToTensor([2])) \
            .setBatchSize(1).setLearningRate(0.2).setMaxEpoch(2) \
            .setDataCacheLevel("DISK_AND_DRAM", 2)

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        estimator.fit(df)

    def test_nnEstimator_fit_with_non_default_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), SeqToTensor([2]))\
            .setBatchSize(4)\
            .setLearningRate(0.01).setMaxEpoch(1) \
            .setFeaturesCol("abcd").setLabelCol("xyz").setPredictionCol("tt")

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("abcd", ArrayType(DoubleType(), False), False),
            StructField("xyz", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnModel = estimator.fit(df)

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.select("abcd", "xyz", "tt").count() == 4

    def test_nnEstimator_fit_with_different_OptimMethods(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), SeqToTensor([2]))\
            .setBatchSize(4)\
            .setLearningRate(0.01).setMaxEpoch(1) \
            .setPredictionCol("tt")

        df = self.get_estimator_df()
        for opt in [SGD(learningrate=1e-3, learningrate_decay=0.0,),
                    Adam(), LBFGS(), Adagrad(), Adadelta()]:
            nnModel = estimator.setOptimMethod(opt).fit(df)
            res = nnModel.transform(df)
            assert type(res).__name__ == 'DataFrame'
            assert res.select("features", "label", "tt").count() == 4

    def test_nnEstimator_fit_with_adam_lr_schedile(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        df = self.get_estimator_df()
        nnModel = NNEstimator(model, criterion, SeqToTensor([2]), SeqToTensor([2])) \
            .setBatchSize(4) \
            .setLearningRate(0.01).setMaxEpoch(1) \
            .setPredictionCol("tt") \
            .setOptimMethod(KAdam(
                schedule=Plateau("Loss", factor=0.1, patience=2, mode="min", epsilon=0.01,
                                 cooldown=0, min_lr=1e-15))) \
            .fit(df)
        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'

    def test_nnEstimator_create_with_feature_size(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, [2], [2])\
            .setBatchSize(4).setLearningRate(0.2).setMaxEpoch(1)

        df = self.get_estimator_df()
        nnModel = estimator.fit(df)
        assert nnModel.getBatchSize() == 4

    def test_nnEstimator_fit_with_train_val_summary(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
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
        df = self.sqlContext.createDataFrame(data, schema)
        val_df = self.sqlContext.createDataFrame(val_data, schema)

        tmp_dir = tempfile.mkdtemp()
        train_summary = TrainSummary(log_dir=tmp_dir, app_name="estTest")
        train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))
        val_summary = ValidationSummary(log_dir=tmp_dir, app_name="estTest")
        estimator = NNEstimator(model, criterion, SeqToTensor([2]), SeqToTensor([2]))\
            .setBatchSize(4) \
            .setMaxEpoch(5) \
            .setTrainSummary(train_summary)
        assert (estimator.getValidation() is None)
        estimator.setValidation(EveryEpoch(), val_df, [MAE()], 2) \
            .setValidationSummary(val_summary)
        assert (estimator.getValidation() is not None)

        nnModel = estimator.fit(df)
        res = nnModel.transform(df)
        lr_result = train_summary.read_scalar("LearningRate")
        mae_result = val_summary.read_scalar("MAE")
        assert isinstance(estimator.getTrainSummary(), TrainSummary)
        assert type(res).__name__ == 'DataFrame'
        assert len(lr_result) == 5
        assert len(mae_result) == 4

    def test_NNEstimator_checkpoint(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        df = self.get_estimator_df()
        try:
            tmp_dir = tempfile.mkdtemp()
            estimator = NNEstimator(model, criterion).setMaxEpoch(5)\
                .setBatchSize(4)\
                .setCheckpoint(tmp_dir, EveryEpoch(), False)

            checkpoint_config = estimator.getCheckpoint()
            assert checkpoint_config[0] == tmp_dir
            assert "EveryEpoch" in str(checkpoint_config)
            assert checkpoint_config[2] is False

            estimator.fit(df)
            assert len(os.listdir(tmp_dir)) > 0
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_NNEstimator_multi_input(self):
        zx1 = ZLayer.Input(shape=(1, ))
        zx2 = ZLayer.Input(shape=(1, ))
        zz = ZLayer.merge([zx1, zx2], mode="concat")
        zy = ZLayer.Dense(2)(zz)
        zmodel = ZModel([zx1, zx2], zy)

        criterion = MSECriterion()
        df = self.get_estimator_df()
        estimator = NNEstimator(zmodel, criterion, [[1], [1]]).setMaxEpoch(5) \
            .setBatchSize(4)
        nnmodel = estimator.fit(df)
        nnmodel.transform(df).collect()

    def test_NNEstimator_works_with_VectorAssembler_multi_input(self):
        if self.sc.version.startswith("2"):
            from pyspark.ml.linalg import Vectors
            from pyspark.ml.feature import VectorAssembler
            from pyspark.sql import SparkSession

            spark = SparkSession \
                .builder \
                .getOrCreate()

            df = spark.createDataFrame(
                [(1, 35, 109.0, Vectors.dense([2.0, 5.0, 0.5, 0.5]), 1.0),
                 (2, 58, 2998.0, Vectors.dense([4.0, 10.0, 0.5, 0.5]), 2.0),
                 (3, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0)],
                ["user", "age", "income", "history", "label"])

            assembler = VectorAssembler(
                inputCols=["user", "age", "income", "history"],
                outputCol="features")

            df = assembler.transform(df)

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
            classifier = NNClassifier(zmodel, criterion, [[1], [2], [2, 2]]) \
                .setOptimMethod(Adam()) \
                .setLearningRate(0.1) \
                .setBatchSize(2) \
                .setMaxEpoch(10)

            nnClassifierModel = classifier.fit(df)
            print(nnClassifierModel.getBatchSize())
            res = nnClassifierModel.transform(df).collect()

    def test_NNModel_transform_with_nonDefault_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        nnModel = NNModel(model, SeqToTensor([2]))\
            .setFeaturesCol("abcd").setPredictionCol("dcba")

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("abcd", ArrayType(DoubleType(), False), False),
            StructField("xyz", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.select("abcd", "dcba").count() == 4

    def test_nnModel_set_Preprocessing(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion, [2], [2])\
            .setBatchSize(4).setLearningRate(0.2).setMaxEpoch(1)

        df = self.get_estimator_df()
        nnModel = estimator.fit(df)
        newTransformer = ChainedPreprocessing([SeqToTensor([2]), TensorToSample()])
        nnModel.setSamplePreprocessing(newTransformer)

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.count() == 4

    def test_NNModel_save_load_BigDL_model(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion).setMaxEpoch(1).setBatchSize(4)

        df = self.get_estimator_df()
        nnModel = estimator.fit(df)
        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            nnModel.model.save(modelPath)
            loaded_model = Model.load(modelPath)
            resultDF = NNModel(loaded_model).transform(df)
            assert resultDF.count() == 4
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_NNModel_save_load(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator(model, criterion).setMaxEpoch(1).setBatchSize(4)

        df = self.get_estimator_df()
        nnModel = estimator.fit(df)
        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            nnModel.save(modelPath)
            loaded_model = NNModel.load(modelPath)
            assert loaded_model.transform(df).count() == 4
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_nnclassifier_fit_nnclassifiermodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2])) \
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)

        df = self.get_classifier_df()
        nnClassifierModel = classifier.fit(df)
        assert(isinstance(nnClassifierModel, NNClassifierModel))
        res = nnClassifierModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("nnClassifierModelDF")
        results = self.sqlContext.table("nnClassifierModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert row_label == row_prediction

    def test_nnclassifier_fit_with_Sigmoid(self):
        model = Sequential().add(Linear(2, 1)).add(Sigmoid())
        criterion = BCECriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2])) \
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)

        data = self.sc.parallelize([
            ((2.0, 1.0), 0.0),
            ((1.0, 2.0), 1.0),
            ((2.0, 1.0), 0.0),
            ((1.0, 2.0), 1.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnClassifierModel = classifier.fit(df)
        assert(isinstance(nnClassifierModel, NNClassifierModel))
        res = nnClassifierModel.transform(df)
        res.registerTempTable("nnClassifierModelDF")
        results = self.sqlContext.table("nnClassifierModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert row_label == row_prediction

    def test_nnclassifierModel_set_Preprocessing(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2])) \
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(1)

        df = self.get_classifier_df()
        nnClassifierModel = classifier.fit(df)

        newTransformer = ChainedPreprocessing([SeqToTensor([2]), TensorToSample()])
        nnClassifierModel.setSamplePreprocessing(newTransformer)

        res = nnClassifierModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        assert res.count() == 4

    def test_nnclassifier_create_with_size_fit_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, [2]) \
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)

        df = self.get_classifier_df()
        nnClassifierModel = classifier.fit(df)

        res = nnClassifierModel.transform(df)
        assert type(res).__name__ == 'DataFrame'

    def test_nnclassifier_fit_different_optimMethods(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2]))\
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(1)

        df = self.get_classifier_df()
        for opt in [Adam(), SGD(learningrate=1e-2, learningrate_decay=1e-6,),
                    LBFGS(), Adagrad(), Adadelta()]:
            nnClassifierModel = classifier.setOptimMethod(opt).fit(df)
            res = nnClassifierModel.transform(df)
            res.collect()
            assert type(res).__name__ == 'DataFrame'

    def test_nnClassifier_fit_with_train_val_summary(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        val_data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        val_df = self.sqlContext.createDataFrame(val_data, schema)

        tmp_dir = tempfile.mkdtemp()
        train_summary = TrainSummary(log_dir=tmp_dir, app_name="nnTest")
        train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))
        val_summary = ValidationSummary(log_dir=tmp_dir, app_name="nnTest")

        classfier = NNClassifier(model, criterion, SeqToTensor([2]))\
            .setBatchSize(4) \
            .setTrainSummary(train_summary).setMaxEpoch(5) \
            .setValidation(EveryEpoch(), val_df, [Top1Accuracy()], 2) \
            .setValidationSummary(val_summary)

        nnModel = classfier.fit(df)
        res = nnModel.transform(df)
        lr_result = train_summary.read_scalar("LearningRate")
        top1_result = val_summary.read_scalar("Top1Accuracy")

        assert isinstance(classfier.getTrainSummary(), TrainSummary)
        assert type(res).__name__ == 'DataFrame'
        assert len(lr_result) == 5
        assert len(top1_result) == 4

    def test_nnclassifier_in_pipeline(self):

        if self.sc.version.startswith("1"):
            from pyspark.mllib.linalg import Vectors

            df = self.sqlContext.createDataFrame(
                [(Vectors.dense([2.0, 1.0]), 1.0),
                 (Vectors.dense([1.0, 2.0]), 2.0),
                 (Vectors.dense([2.0, 1.0]), 1.0),
                 (Vectors.dense([1.0, 2.0]), 2.0),
                 ], ["features", "label"])

            scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaled")
            model = Sequential().add(Linear(2, 2))
            criterion = ClassNLLCriterion()
            classifier = NNClassifier(model, criterion)\
                .setBatchSize(4) \
                .setLearningRate(0.01).setMaxEpoch(1).setFeaturesCol("scaled")

            pipeline = Pipeline(stages=[scaler, classifier])

            pipelineModel = pipeline.fit(df)

            res = pipelineModel.transform(df)
            assert type(res).__name__ == 'DataFrame'
        # TODO: Add test for ML Vector once infra is ready.

    def test_NNClassifierModel_save_load_BigDL_model(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        classifier = NNClassifier(model, criterion).setMaxEpoch(1).setBatchSize(4)

        df = self.get_classifier_df()
        nnClassifierModel = classifier.fit(df)
        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            nnClassifierModel.model.save(modelPath)
            loaded_model = Model.load(modelPath)
            resultDF = NNClassifierModel(loaded_model).transform(df)
            assert resultDF.count() == 4
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_NNClassifierModel_save_load(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, [2]).setMaxEpoch(1).setBatchSize(4)

        df = self.get_classifier_df()
        nnClassifierModel = classifier.fit(df)
        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            nnClassifierModel.save(modelPath)
            loaded_model = NNClassifierModel.load(modelPath)
            assert (isinstance(loaded_model, NNClassifierModel))
            assert loaded_model.transform(df).count() == 4
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_input_node_of_tfnet_from_session(self):
        import tensorflow as tff
        input1 = tff.placeholder(dtype=tff.float32, shape=(None, 2))
        input2 = tff.placeholder(dtype=tff.float32, shape=(None, 2))
        hidden = tff.layers.dense(input1, 4)
        output = tff.layers.dense(hidden, 1)
        sess = tff.Session()
        sess.run(tff.global_variables_initializer())
        tmp_dir = tempfile.mkdtemp()
        modelPath = os.path.join(tmp_dir, "model")
        raised_error = False
        try:
            export_tf(sess, modelPath, inputs=[input1, input2], outputs=[output])
        except ValueError as v:
            assert (((str(v)).find((input2.name)[0:-2])) != -1)
            raised_error = True
        finally:
            try:
                shutil.rmtree(modelPath)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

        if not raised_error:
            raise ValueError("we do not find this error, test failed")


if __name__ == "__main__":
    pytest.main()
