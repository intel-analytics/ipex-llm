/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.ppml.fl.example.benchmark

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.fl.algorithms.FGBoostRegression
import com.intel.analytics.bigdl.ppml.fl.FLContext
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.XGBoostFormatValidator
import com.intel.analytics.bigdl.ppml.fl.utils.{DataFrameUtils, TimingSupportive}
import scopt.OptionParser

import scala.io.Source
import scala.util.Random


object FGBoostRealDataBenchmark extends TimingSupportive {
  case class Params(trainPath: String = null,
                    testPath: String = null,
                    dataSize: Int = 1,
                    numRound: Int = 10)
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Params]("Text Classification Example") {
      opt[String]("trainPath")
        .text("data size")
        .action((x, params) => params.copy(trainPath = x))
        .required()
      opt[String]("testPath")
        .text("data size")
        .action((x, params) => params.copy(testPath = x))
        .required()
      opt[Int]("dataSize")
        .text("data dimension")
        .action((x, params) => params.copy(dataSize = x))
      opt[Int]("numRound")
        .text("boosting round numer")
        .action((x, params) => params.copy(numRound = x))
    }

    val param = parser.parse(args, Params()).get
    val spark = FLContext.getSparkSession()
    val trainDf = spark.read.option("header", "true").csv(param.trainPath)
    val testDf = spark.read.option("header", "true").csv(param.testPath)


    // Data copy for simulating large dataset
    val trainDataArray = DataFrameUtils.toTensorArray(trainDf.drop("SalePrice"))
    val trainLabelArray = DataFrameUtils.toArray(trainDf, "SalePrice")

    var trainDataStacked = Array[Tensor[Float]]()
    var trainLabelStacked = Array[Float]()
    (0 until param.dataSize).foreach(_ => {
      trainDataStacked = trainDataStacked ++ trainDataArray
      trainLabelStacked = trainLabelStacked ++ trainLabelArray
    })

    FLContext.initFLContext(1)
    val fGBoostRegression = new FGBoostRegression(
      learningRate = 0.1f, maxDepth = 7, minChildSize = 5)
    fGBoostRegression.fit(
      trainDataStacked,
      trainLabelStacked,
      param.numRound)

    val fGBoostResult = fGBoostRegression.predict(DataFrameUtils.toTensorArray(testDf))
      .map(tensor => tensor.value())
      .map(math.exp(_))

  }
}
