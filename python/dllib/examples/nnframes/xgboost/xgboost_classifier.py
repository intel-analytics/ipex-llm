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
from optparse import OptionParser
import numpy as np
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.nn.layer import *
from bigdl.dllib.optim.optimizer import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.common import *
from bigdl.dllib.feature.image import *
from bigdl.dllib.keras import layers as ZLayer
from bigdl.dllib.keras.models import Model as ZModel
from bigdl.dllib.keras.optimizers import Adam as KAdam
from bigdl.dllib.nnframes import *
from bigdl.dllib.utils.tf import *
import csv
import errno


def process(filepath, demo):
    sparkConf = init_spark_conf().setAppName("testXGBClassifier")
    sparkConf = sparkConf.set("xgboost.spark.ignoreSsl", True)
    sc = init_nncontext(sparkConf)
    sqlContext = SQLContext(sc)
    if demo:
        data = sc.parallelize([
            (1.0, 2.0, 3.0, 4.0, 7.0, 1),
            (1.0, 3.0, 8.0, 2.0, 5.0, 0),
            (2.0, 3.4, 5.0, 2.0, 4.0, 1)
        ])
        N = 6
        train_data = data
        test_data = data
    else:
        dataset = np.loadtxt(filepath, delimiter=',')
        print(type(dataset))
        M, N = dataset.shape
        train_X = dataset[:(int)(0.8 * M), :]
        test_X = dataset[(int)(0.8 * M):, :]
        train_data = sc.parallelize(train_X.tolist())
        test_data = sc.parallelize(test_X.tolist())
    columns = ["f" + str(i) for i in range(1, N)]
    features = ["f" + str(i) for i in range(1, N)]

    columns.append("label")
    df1 = train_data.toDF(columns)
    vecasembler = VectorAssembler(inputCols=features, outputCol="features")
    traindf = vecasembler.transform(df1).select("features", "label").cache()
    traindf.show()

    df2 = test_data.toDF(columns)
    testdf = vecasembler.transform(df2).select("features", "label").cache()
    testdf.show()
    xgbCf0 = XGBClassifier()
    xgbCf0.setNthread(1)
    xgbCf0.setNumRound(10)
    xgbCf0.setMissing(0)
    xgbmodel = xgbCf0.fit(traindf)
    xgbmodel.setFeaturesCol("features")
    y0 = xgbmodel.transform(testdf)
    print(y0)
    y0.show()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file-path", type=str, dest="data_path",
                      default=".", help="Path where data are stored")
    parser.add_option("-d", "--demo", action="store_true", dest="demo", default=False)
    parser.add_option("-m", "--master", type=str, dest="the master choice")
    (option, args) = parser.parse_args(sys.argv)

    if option.data_path is None:
        errno("data path is not specified")
    datapath = option.data_path
    process(datapath, option.demo)
