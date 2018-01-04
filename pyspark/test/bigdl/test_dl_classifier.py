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

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.models.ml_pipeline.dl_classifier import *
from pyspark.sql.types import *
from pyspark.context import SparkContext
import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestDLClassifer():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[1]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        init_engine()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_all_set_get_methods(self):
        """ run tests for all the set and get methods involved in DLEstimator, DLModel,
            DLClassifier, DLClassifierModel
        """

        '''
        use linear model and MSE criterion to test the DLEstimator and DLModel
        '''
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()

        '''
        initialize a DLEstimator to test setter, getter for
        batchSize, maxEpoch, learningRate in DLEstimator
        '''
        estimator = DLEstimator(model=linear_model, criterion=mse_criterion,
                                feature_size=[2], label_size=[2])

        # check the set and get methods for batchSize
        assert estimator.setBatchSize(30).getBatchSize() == 30
        # check the set and get methods for maxEpoch
        assert estimator.setMaxEpoch(40).getMaxEpoch() == 40
        # check the set and get methods for learningRate
        assert estimator.setLearningRate(1e-4).getLearningRate() == 1e-4

        '''
        initialize a DLModel to test setter, getter for featureSize, batchSize in DLModel
        '''
        dl_model = DLClassifierModel(model=linear_model, featureSize=[1])

        # check the set and get methods for featureSize in DLModel
        assert dl_model.setFeatureSize([2]).getFeatureSize() == [2]
        # check the set and get methods for batchSize in DLModel
        assert dl_model.setBatchSize((20)).getBatchSize() == 20

        '''
        use linear model and ClassNLL criterion to test the DLClassifier and DLClassifierModel
        '''
        linear_model = Sequential().add(Linear(2, 2))
        classNLL_criterion = ClassNLLCriterion()

        '''
        initialize a DLClassifier to test setter, getter for
        batchSize, maxEpoch, learningRate in DLClassifier
        '''
        classifier = DLClassifier(model=linear_model, criterion=classNLL_criterion,
                                  feature_size=[2])

        # check the set and get methods for batchSize
        assert classifier.setBatchSize(20).getBatchSize() == 20
        # check the set and get methods for maxEpoch
        assert classifier.setMaxEpoch(50).getMaxEpoch() == 50
        # check the set and get methods for learningRate
        assert classifier.setLearningRate(1e-5).getLearningRate() == 1e-5
        '''
        initialize a DLClassifierModel to test setter, getter
        for featureSize, batchSize in DLClassifierModel
        '''
        dl_classifier_model = DLClassifierModel(model=linear_model, featureSize=[1])

        # check the set and get methods for featureSize
        assert dl_classifier_model.setFeatureSize([2]).getFeatureSize() == [2]

        # check the set and get methods for batchSize
        assert dl_classifier_model.setBatchSize((20)).getBatchSize() == 20

    def test_dlestimator_fit_dlmodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = MSECriterion()
        estimator = DLEstimator(model, criterion, [2], [2]).setBatchSize(4)\
            .setLearningRate(0.01).setMaxEpoch(1000)

        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        dlModel = estimator.fit(df)

        res = dlModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("dlModelDF")  # Compatible with spark 1.6
        results = self.sqlContext.table("dlModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert_allclose(row_label[0], row_prediction[0], atol=0, rtol=1e-1)
            assert_allclose(row_label[1], row_prediction[1], atol=0, rtol=1e-1)

    def test_dlclassifier_fit_dlclassifiermodel_transform(self):
        model = Sequential().add(Linear(2, 2))
        criterion = ClassNLLCriterion()
        classifier = DLClassifier(model, criterion, [2]).setBatchSize(4)\
            .setLearningRate(0.01).setMaxEpoch(1000)
        data = self.sc.parallelize([
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0),
            ((2.0, 1.0), 1.0),
            ((1.0, 2.0), 2.0)])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", DoubleType(), False)])
        df = self.sqlContext.createDataFrame(data, schema)
        dlClassifierModel = classifier.fit(df)

        res = dlClassifierModel.transform(df)
        assert type(res).__name__ == 'DataFrame'
        res.registerTempTable("dlClassifierModelDF")
        results = self.sqlContext.table("dlClassifierModelDF")

        count = results.rdd.count()
        data = results.rdd.collect()

        for i in range(count):
            row_label = data[i][1]
            row_prediction = data[i][2]
            assert row_label == row_prediction

if __name__ == "__main__":
    pytest.main()
