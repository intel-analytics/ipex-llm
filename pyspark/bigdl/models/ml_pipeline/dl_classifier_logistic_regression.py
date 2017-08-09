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
# Still in experimental stage!

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.ml.dl_classifier import *
from pyspark.sql.types import *

if __name__ == "__main__":

    sc = SparkContext(appName="DLClassifierLogisticRegression", conf=create_spark_conf().setMaster("local[1]"))
    sqlContext = SQLContext(sc)
    init_engine()
    model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
    criterion = ClassNLLCriterion()
    estimator = DLClassifier(model, criterion, [2]).setBatchSize(4).setMaxEpoch(10)
    data = sc.parallelize([
        ((0.0, 1.0), [1.0]),
        ((1.0, 0.0), [2.0]),
        ((0.0, 1.0), [1.0]),
        ((1.0, 0.0), [2.0])])

    schema = StructType([
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("label", ArrayType(DoubleType(), False), False)])
    df = sqlContext.createDataFrame(data, schema)
    dlModel = estimator.fit(df)
    dlModel.transform(df).show(False)
    sc.stop()
