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
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import SGD, Adam, LBFGS, Adagrad, Adadelta
from bigdl.util.common import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *


class TestNNClassifer():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[1]").setAppName("testNNClassifer")
        self.sc = get_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_nnEstimator_construct_with_sample_transformer(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        estimator = NNEstimator(
            linear_model, mse_criterion, FeatureLabelPreprocessing(SeqToTensor([2]), SeqToTensor([2]))
        ).setBatchSize(4).setMaxEpoch(1)
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnModel = estimator.fit(df)

        res = nnModel.transform(df)
        assert type(res).__name__ == 'DataFrame'

    def test_all_set_get_methods(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()

        estimator = NNEstimator.create(
            linear_model, mse_criterion, SeqToTensor([2]), SeqToTensor([2])
        )
        assert estimator.setBatchSize(30).getBatchSize() == 30
        assert estimator.setMaxEpoch(40).getMaxEpoch() == 40
        assert estimator.setLearningRate(1e-4).getLearningRate() == 1e-4
        assert estimator.setFeaturesCol("abcd").getFeaturesCol() == "abcd"
        assert estimator.setLabelCol("xyz").getLabelCol() == "xyz"
        assert isinstance(estimator.setOptimMethod(Adam()).getOptimMethod(), Adam)

        nn_model = NNModel.create(linear_model, SeqToTensor([2]))
        assert nn_model.setBatchSize(20).getBatchSize() == 20

        linear_model = Sequential().add(Linear(2, 2))
        classNLL_criterion = ClassNLLCriterion()
        classifier = NNClassifier(
            model=linear_model, criterion=classNLL_criterion,
            feature_preprocessing = SeqToTensor([2])
        )
        assert classifier.setBatchSize(20).getBatchSize() == 20
        assert classifier.setMaxEpoch(50).getMaxEpoch() == 50
        assert classifier.setLearningRate(1e-5).getLearningRate() == 1e-5

        nn_classifier_model = NNClassifierModel(linear_model, SeqToTensor([2]))
        assert nn_classifier_model.setBatchSize((20)).getBatchSize() == 20

    def test_nnEstimator_fit_nnmodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator.create(
            model, criterion, SeqToTensor([2]), ArrayToTensor([2])
        ).setBatchSize(4).setLearningRate(0.2).setMaxEpoch(40)

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
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

    def test_nnEstimator_fit_with_non_default_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = NNEstimator.create(
            model, criterion, SeqToTensor([2]), SeqToTensor([2])
        ).setBatchSize(4)\
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
        estimator = NNEstimator.create(
            model, criterion, SeqToTensor([2]), SeqToTensor([2]))\
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

        for opt in [SGD(learningrate=1e-3, learningrate_decay=0.0,),
                    Adam(), LBFGS(), Adagrad(), Adadelta()]:
            nnModel = estimator.setOptimMethod(opt).fit(df)
            res = nnModel.transform(df)
            assert type(res).__name__ == 'DataFrame'
            assert res.select("abcd", "xyz", "tt").count() == 4

    def test_NNModel_transform_with_nonDefault_featureCol(self):
        model = Sequential().add(Linear(2, 2))
        nnModel = NNModel.create(model, SeqToTensor([2]))\
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

    def test_nnclassifier_fit_nnclassifiermodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2])) \
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(40)
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        nnClassifierModel = classifier.fit(df)

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

    def test_nnclassifier_fit_different_optimMethods(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = NNClassifier(model, criterion, SeqToTensor([2]))\
            .setBatchSize(4) \
            .setLearningRate(0.2).setMaxEpoch(1)
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)

        for opt in [Adam(), SGD(learningrate=1e-2, learningrate_decay=1e-6,),
                    LBFGS(), Adagrad(), Adadelta()]:
            nnClassifierModel = classifier.setOptimMethod(opt).fit(df)
            res = nnClassifierModel.transform(df)
            res.collect()
            assert type(res).__name__ == 'DataFrame'

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
            classifier = NNClassifier(model, criterion, MLlibVectorToTensor([2]))\
                .setBatchSize(4) \
                .setLearningRate(0.01).setMaxEpoch(1).setFeaturesCol("scaled")

            pipeline = Pipeline(stages=[scaler, classifier])

            pipelineModel = pipeline.fit(df)

            res = pipelineModel.transform(df)
            assert type(res).__name__ == 'DataFrame'
        #TODO: Add test for ML Vector once infra is ready.

if __name__ == "__main__":
    pytest.main()
