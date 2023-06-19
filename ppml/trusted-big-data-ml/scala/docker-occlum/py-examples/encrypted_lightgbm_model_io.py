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

from bigdl.ppml.ppml_context import *
from bigdl.ppml.kms.utils.kms_argument_parser import KmsArgumentParser
from pyspark.sql.types import *
from pyspark.ml.feature import *
from synapse.ml.lightgbm import LightGBMClassifier

args = KmsArgumentParser().get_arg_dict()
sc = PPMLContext('pyspark-lightgbm', args)

schema = StructType([StructField("sepal length", DoubleType(), True),
                     StructField("sepal width", DoubleType(), True),
                     StructField("petal length", DoubleType(), True),
                     StructField("petal width", DoubleType(), True),
                     StructField("class", StringType(), True)])

# read data for training
df = sc.read(args["input_encrypt_mode"]).schema(schema).csv(args["input_path"])

stringIndexer = StringIndexer() \
    .setInputCol("class") \
    .setOutputCol("classIndex") \
    .fit(df)
labelTransformed = stringIndexer.transform(df).drop("class")
vectorAssembler = VectorAssembler() \
    .setInputCols(["sepal length", "sepal width", "petal length", "petal width"]) \
    .setHandleInvalid("skip") \
    .setOutputCol("features")
dfinput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")
(train, test) = dfinput.randomSplit([0.8, 0.2])

# fit a classification model
classificationModel = LightGBMClassifier(featuresCol = "features",
                                         labelCol = "classIndex",
                                         numIterations = 100,
                                         numLeaves = 10,
                                         maxDepth = 6,
                                         lambdaL1 = 0.01,
                                         lambdaL2 = 0.01,
                                         baggingFreq = 5,
                                         maxBin = 255).fit(train)

# save the trained model in ciphertext
sc.saveLightGBMModel(classificationModel, args["output_path"], args["output_encrypt_mode"])

# load the encrypted model and use it to predict
reloadedModel = sc.loadLightGBMClassificationModel(args["output_path"], args["output_encrypt_mode"])
predictions = reloadedModel.transform(test)
predictions.show(10)