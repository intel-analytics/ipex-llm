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

package com.intel.analytics.bigdl.ppml.fl.example

import com.intel.analytics.bigdl.ppml.fl.FLContext
import com.intel.analytics.bigdl.ppml.fl.algorithms.{PSI, VFLLogisticRegression}
import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fl.utils.TensorUtils
import org.apache.spark.sql.DataFrame
import scopt.OptionParser

import collection.JavaConverters._


object VFLLogisticRegression extends DebugLogger{
  def getData(pSI: PSI, dataPath: String, rowKeyName: String): (DataFrame, DataFrame, DataFrame) = {
    val salt = pSI.getSalt()

    val spark = FLContext.getSparkSession()
    val df = spark.read.option("header", "true").csv(dataPath)
    val intersectionDf = pSI.uploadSetAndDownloadIntersectionDataFrame(df, salt, rowKeyName)
    val (trainDf, valDf) = ExampleUtils.splitDataFrameToTrainVal(intersectionDf)
    val testDf = trainDf.drop("Outcome")
    trainDf.show()
    testDf.show()
    (trainDf, valDf, testDf)
  }

  def main(args: Array[String]): Unit = {
    case class Params(clientId: Int = 0,
                      dataPath: String = null,
                      rowKeyName: String = "ID",
                      learningRate: Float = 0.005f,
                      batchSize: Int = 4)
    val parser: OptionParser[Params] = new OptionParser[Params]("VFL Logistic Regression") {
      opt[Int]('c', "clientId")
        .text("data path to load")
        .action((x, params) => params.copy(clientId = x))
        .required()
      opt[String]('d', "dataPath")
        .text("data path to load")
        .action((x, params) => params.copy(dataPath = x))
        .required()
      opt[String]('r', "rowKeyName")
        .text("row key name of data")
        .action((x, params) => params.copy(rowKeyName = x))
      opt[String]('l', "learningRate")
        .text("learning rate of training")
        .action((x, params) => params.copy(learningRate = x.toFloat))
      opt[String]('b', "batchSize")
        .text("batchsize of training")
        .action((x, params) => params.copy(batchSize = x.toInt))

    }
    val argv = parser.parse(args, Params()).head
    val dataPath = argv.dataPath
    val rowKeyName = argv.rowKeyName
    val learningRate = argv.learningRate

    /**
     * Usage of BigDL PPML starts from here
     */
    FLContext.initFLContext(argv.clientId)
    val pSI = new PSI()
    val (trainData, valData, testData) = getData(pSI, dataPath, rowKeyName)

    // create LogisticRegression object to train the model


    // Data pipeline from DataFrame to Tensor, and call fit, evaluate, predict
    val (featureColumns, labelColumns) = argv.clientId match {
      case 1 => (Array("Pregnancies", "Glucose", "BloodPressure", "SkinThickness"),
        Array("Outcome"))
      case 2 => (Array("Insulin", "BMI", "DiabetesPedigreeFunction"), null)
      case _ => throw new IllegalArgumentException("clientId only support 1, 2 in this example")
    }
    val xTrain = TensorUtils.fromDataFrame(trainData, featureColumns)
    val xEval = TensorUtils.fromDataFrame(valData, featureColumns)
    val xTest = TensorUtils.fromDataFrame(testData, featureColumns)
    val yTrain = TensorUtils.fromDataFrame(trainData, labelColumns)
    val yEval = TensorUtils.fromDataFrame(valData, labelColumns)

    val lr = new VFLLogisticRegression(featureColumns.length, learningRate)
    lr.fit(xTrain, yTrain, epoch = 5)
    lr.evaluate(xEval, yEval)
    lr.predict(xTest)
  }

}
