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

import pytest
from bigdl.dllib.nncontext import *
from bigdl.dllib.nnframes.tree_model import *
from bigdl.dllib.utils.tf import *
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.functions import udf, array


class TestTreeModel():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testTreeModel")
        self.sc = init_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        assert (self.sc.appName == "testTreeModel")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_XGBClassifierModel_predict(self):
        from sys import platform
        if platform in ("darwin", "win32"):
            return

        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        path = os.path.join(resource_path, "xgbclassifier/")
        modelPath = path + "XGBClassifer.bin"
        filePath = path + "test.csv"
        model = XGBClassifierModel.loadModel(modelPath, 2)
        df = self.sqlContext.read.csv(filePath, sep=",", inferSchema=True, header=True)
        df = df.select(array("age", "gender", "jointime", "star").alias("features")) \
            .withColumn("features", udf(lambda x: DenseVector(x), VectorUDT())("features"))

        model.setFeaturesCol("features")
        predict = model.transform(df)
        assert predict.count() == 14

    def test_XGBClassifier_train(self):
        from sys import platform
        if platform in ("darwin", "win32"):
            return
        path = os.path.join(self.resource_path, "xgbclassifier/")
        modelPath = path + "XGBClassifer.bin"
        filePath = path + "test.csv"

        df = self.sqlContext.read.csv(filePath, sep=",", inferSchema=True, header=True)
        df = df.select(array("age", "gender", "jointime", "star").alias("features"), "label")\
            .withColumn("features", udf(lambda x: DenseVector(x), VectorUDT())("features"))
        params = {"eta": 0.2, "max_depth":4, "max_leaf_nodes": 8, "objective": "binary:logistic",
                  "num_round": 100}
        classifier = XGBClassifier(params)
        xgbmodel = classifier.fit(df)
        xgbmodel.setFeaturesCol("features")
        predicts = xgbmodel.transform(df)
        assert predicts.count() == 14

    def test_XGBRegressor(self):
        from sys import platform
        if platform in ("darwin", "win32"):
            return

        if self.sc.version.startswith("3.1") or self.sc.version.startswith("2.4"):
            data = self.sc.parallelize([
                (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
                (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
                (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
                (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
            ])
            columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label"]
            df = data.toDF(columns)
            from pyspark.ml.feature import VectorAssembler
            vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
            assembledf = vecasembler.transform(df).select("features", "label").cache()
            assembledf.printSchema()
            testdf = vecasembler.transform(df).select("features", "label").cache()
            xgbRf0 = XGBRegressor()
            xgbRf0.setNthread(1)
            xgbRf0.setNumRound(10)
            xgbmodel = xgbRf0.fit(assembledf)
            xgbmodel.save("/tmp/modelfile/")
            xgbmodel.setFeaturesCol("features")
            yxgb = xgbmodel.transform(assembledf)
            model = xgbmodel.load("/tmp/modelfile/")
            model.setFeaturesCol("features")
            y0 = model.transform(assembledf)
            assert (y0.subtract(yxgb).count() == 0)

    def test_LGBMClassifier_fit_transform(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        path = os.path.join(self.resource_path, "xgbclassifier/")
        filePath = path + "test.csv"
        df = self.sqlContext.read.csv(filePath, sep=",", inferSchema=True, header=True)
        df = df.select(array("age", "gender", "jointime", "star").alias("features"), "label")\
            .withColumn("features", udf(lambda x: DenseVector(x), VectorUDT())("features"))
        # input_path = "/Users/guoqiong/intelWork/data/tweet/xgb_processed"
        # df = self.sqlContext.read.parquet(input_path + "/train")
        classifier = LightGBMClassifier()
        classifier.setObjective("binary")
        classifier.setMaxDepth(4)
        classifier.setLearningRate(0.2)
        model = classifier.fit(df)
        predicts = model.transform(df)
        print(predicts.filter(predicts["prediction"] == 1.0).count())
        assert predicts.count() == 14

    def test_LGBMClassifier_param_map(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        path = os.path.join(self.resource_path, "xgbclassifier/")
        filePath = path + "test.csv"
        df = self.sqlContext.read.csv(filePath, sep=",", inferSchema=True, header=True)
        df = df.select(array("age", "gender", "jointime", "star").alias("features"), "label")\
            .withColumn("features", udf(lambda x: DenseVector(x), VectorUDT())("features"))

        # input_path = "/Users/guoqiong/intelWork/data/tweet/xgb_processed"
        # df = self.sqlContext.read.parquet(input_path + "/train")
        parammap = {"boosting_type": "gbdt", "num_leaves": 2, "max_depth": 2, "learning_rate": 0.3,
                    "num_iterations": 10, "bin_construct_sample_cnt": 5, "objective": "binary",
                    "min_split_gain": 0.1, "min_sum_hessian_in_leaf": 0.01, "min_data_in_leaf": 1,
                    "bagging_fraction": 0.4, "bagging_freq": 1, "feature_fraction": 0.4,
                    "lambda_l1": 0.1, "lambda_l2": 0.1, "num_threads": 2,
                    "early_stopping_round": 10, "max_bin": 100}
        classifier = LightGBMClassifier(parammap)
        model = classifier.fit(df)
        predicts = model.transform(df)
        print(predicts.filter(predicts["prediction"] == 1.0).count())
        assert predicts.count() == 14

    def test_LGBMClassifierModel_save_load(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        path = os.path.join(self.resource_path, "xgbclassifier/")
        filePath = path + "test.csv"
        df = self.sqlContext.read.csv(filePath, sep=",", inferSchema=True, header=True)
        df = df.select(array("age", "gender", "jointime", "star").alias("features"), "label") \
            .withColumn("features", udf(lambda x: DenseVector(x), VectorUDT())("features"))
        classifier = LightGBMClassifier()
        classifier.setObjective("binary")
        model = classifier.fit(df)
        predicts = model.transform(df)
        model.saveModel("/tmp/lightgbmClassifier1")
        model1 = LightGBMClassifierModel.loadModel("/tmp/lightgbmClassifier1")
        predicts1 = model1.transform(df)
        assert predicts1.count() == 14

    def test_LGBMRegressor_param_map(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        data = self.sc.parallelize([
            (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
            (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
            (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
            (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668),
            (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
            (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
            (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
            (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
        ])
        columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label"]
        df = data.toDF(columns)
        from pyspark.ml.feature import VectorAssembler
        vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
        assembledf = vecasembler.transform(df).select("features", "label").cache()
        assembledf.printSchema()
        testdf = vecasembler.transform(df).select("features", "label").cache()
        parammap = {"boosting_type": "dart",  "num_leaves": 2, "max_depth": 2, "learning_rate": 0.3,
                    "num_iterations": 10, "bin_construct_sample_cnt": 5, "objective": "huber",
                    "min_split_gain": 0.1, "min_sum_hessian_in_leaf": 0.01, "min_data_in_leaf": 1,
                    "bagging_fraction": 0.4, "bagging_freq": 1, "feature_fraction": 0.4,
                    "lambda_l1": 0.1, "lambda_l2": 0.1, "num_threads": 2,
                    "early_stopping_round": 10, "max_bin": 100}
        regressor = LightGBMRegressor(parammap)
        model = regressor.fit(assembledf)
        predicts = model.transform(assembledf)
        predicts.show()
        assert (predicts.count() == 8)

    def test_LGBMRegressor_train_transform(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        data = self.sc.parallelize([
            (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
            (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
            (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
            (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
        ])
        columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label"]
        df = data.toDF(columns)
        from pyspark.ml.feature import VectorAssembler
        vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
        assembledf = vecasembler.transform(df).select("features", "label").cache()
        assembledf.printSchema()
        testdf = vecasembler.transform(df).select("features", "label").cache()
        regressor = LightGBMRegressor()
        model = regressor.fit(assembledf)
        predicts = model.transform(assembledf)
        predicts.show()
        assert (predicts.count() == 4)

    def test_LGBMRegressorModel_save_load(self):
        if float(self.sc.version[:3]) < 3.1:
            return
        data = self.sc.parallelize([
            (1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0, 3.0, 116.3668),
            (1.0, 3.0, 8.0, 6.0, 5.0, 9.0, 5.0, 6.0, 7.0, 4.0, 116.367),
            (2.0, 1.0, 5.0, 7.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 116.367),
            (2.0, 1.0, 4.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 3.0, 116.3668)
        ])
        columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label"]
        df = data.toDF(columns)
        from pyspark.ml.feature import VectorAssembler
        vecasembler = VectorAssembler(inputCols=columns, outputCol="features")
        df = vecasembler.transform(df).select("features", "label").cache()
        regressor = LightGBMRegressor()
        model = regressor.fit(df)
        predicts = model.transform(df)
        model.saveModel("/tmp/lightgbmRegressor1")
        model1 = LightGBMRegressorModel.loadModel("/tmp/lightgbmRegressor1")
        predicts1 = model1.transform(df)
        assert (predicts1.count() == 4)


if __name__ == "__main__":
    pytest.main([__file__])
