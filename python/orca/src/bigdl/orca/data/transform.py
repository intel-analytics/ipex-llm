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
from pyspark.ml.feature import MinMaxScaler as SparkMinMaxScaler
from pyspark.ml.feature import VectorAssembler as SparkVectorAssembler
from pyspark.ml.feature import StringIndexer as SparkStringIndexer
from pyspark.ml import Pipeline as SparkPipeline
from bigdl.dllib.utils.log4Error import *
from bigdl.orca.data import SparkXShards
import uuid


class MinMaxScaler:
    def __init__(self, min=0.0, max=1.0, inputCol=None, outputCol=None):
        self.min = min
        self.max = max
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.scaler = None
        self.scalerModel = None
        if inputCol:
            self.__createScaler__()

    def __createScaler__(self):
        invalidInputError(self.inputCol, "inputColumn cannot be empty")
        invalidInputError(self.outputCol, "outputColumn cannot be empty")
        if len(self.inputCol) > 1:
            vecOutputCol = str(uuid.uuid1()) + "x_vec"
            assembler = SparkVectorAssembler(inputCols=self.inputCol, outputCol=vecOutputCol)
            scaler = SparkMinMaxScaler(min=self.min, max=self.max,
                                       inputCol=vecOutputCol, outputCol=self.outputCol)
            self.scaler = SparkPipeline(stages=[assembler, scaler])
        else:
            self.scaler = SparkMinMaxScaler(min=self.min, max=self.max,
                                            inputCol=self.inputCol, outputCol=self.outputCol)

    def setInputOutputCol(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.__createScaler__()

    def fit_transform(self, shard):
        df = shard.to_spark_df()
        self.scalerModel = self.scaler.fit(df)
        scaledData = self.scalerModel.transform(df)
        data_shards = SparkXShards.from_sparkdf(scaledData)
        return data_shards

    def transform(self, shard):
        invalidInputError(self.scalerModel, "Please call fit_transform first")
        df = shard.to_spark_df()
        scaledData = self.scalerModel.transform(df)
        data_shards = SparkXShards.from_sparkdf(scaledData)
        return data_shards

class LabelEncode:
    def __init__(self, inputCol=None, outputCol=None):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.indexer = None
        self.indexModel = None
        if inputCol:
            self.__createIndexer__()

    def setInputOutputCol(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.__createIndexer__()

    def __createIndexer__(self):
        invalidInputError(self.inputCol, "inputColumn cannot be empty")
        invalidInputError(self.outputCol, "outputColumn cannot be empty")
        self.indexer = SparkStringIndexer(inputCol=self.inputCol, outputCol=self.outputCol)

    def fit_transform(self, shard):
        df = shard.to_spark_df()
        self.indexModel = self.indexer.fit(df)
        indexedData = self.indexModel.transform(df)
        data_shards = SparkXShards.from_sparkdf(indexedData)
        return data_shards

    def transform(self, shard):
        invalidInputError(self.indexModel, "Please call fit_transform first")
        df = shard.to_spark_df()
        scaledData = self.indexModel.transform(df)
        data_shards = SparkXShards.from_sparkdf(scaledData)
        return data_shards
