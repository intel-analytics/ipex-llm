/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.fgboost

import com.intel.analytics.bigdl.ppml.algorithms.vfl.FGBoostRegression
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fgboost.common.{XGBoostFormatNode, XGBoostFormatSerializer, XGBoostFormatValidator}
import com.intel.analytics.bigdl.ppml.utils.{MockClient, TmpUtils}
import com.intel.analytics.bigdl.ppml.{FLContext, FLServer}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.log4j.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

class CorrectnessSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger {
  val logger = LogManager.getLogger(getClass)
  var xGBoostResults: Array[Double] = null
  var xgBoostFormatNodes: Array[XGBoostFormatNode] = null
  before {
    val dataPath = getClass.getClassLoader.getResource("house-prices-train.csv").getPath
    val testPath = getClass.getClassLoader.getResource("house-prices-test.csv").getPath
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val testSources = Source.fromFile(testPath, "utf-8").getLines()
    val rowkeyName = "Id"
    val labelName = "SalePrice"
    val (trainFeatures, testFeatures, trainLabels) =
      TmpUtils.preprocessing(sources, testSources, rowkeyName, labelName)
    val trainFeatureArray = trainFeatures.map(tensor => tensor.toArray()).flatten
    val testFeatureArray = testFeatures.map(tensor => tensor.toArray()).flatten
    val trainDMat = new DMatrix(trainFeatureArray, trainFeatures.length,
      trainFeatureArray.length / trainFeatures.length, 0.0f)
    val testDMat = new DMatrix(testFeatureArray, testFeatures.length,
      testFeatureArray.length / testFeatures.length, 0.0f)
    trainDMat.setLabel(trainLabels)
    val params = Map("eta" -> 0.1, "max_depth" -> 7, "objective" -> "reg:squarederror",
    "min_child_weight" -> 5)
    val booster = XGBoost.train(trainDMat, params, 100)
    xGBoostResults = booster.predict(testDMat).flatten.map(math.exp(_))
    val treeStr = booster.getModelDump(featureMap = null, withStats = false, format = "json")
    // XGBoost model dump will get 100(number of boost round) trees, we validate each one
    xgBoostFormatNodes = treeStr.map(singleTree => {
      XGBoostFormatSerializer(singleTree)
    })

    xGBoostResults
  }
  // House pricing dataset compared with xgboost training and prediction result
  //TODO: use DataFrame API to do the same validation
  "FGBoost Correctness single party" should "work" in {
//    val spark = FLContext.getSparkSession()
//    var df = spark.read.option("header", "true")
//      .csv(getClass.getClassLoader.getResource("house-prices-train.csv").getPath)
//    df = DataFrameUtils.fillNA(df)
//
//    val categoricalFeatures = Array(
//      "MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities",
//      "LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle",
//      "RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond",
//      "Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating",
//      "HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu",
//      "GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence",
//      "MiscFeature","SaleType","SaleCondition"
//    )
//    val categoricalValueMap = DataFrameUtils.getCategoricalValueMap(
//      df, featureColumns = categoricalFeatures)
//    categoricalFeatures.foreach(colName => {
//      df = DataFrameUtils.stringColumnToIndex(df, colName, categoricalValueMap(colName))
//    })
//    val numericalFeatures = (df.columns.toSet -- categoricalFeatures.toSet).toArray
//    df = DataFrameUtils.dataFrameFloatCasting(df, numericalFeatures)
//    df.show()
//    val (feature, label) = DataFrameUtils.dataFrameToMatrix(
//      df, categoricalValueMap, Array("SalePrice"))
//    val normalizedLabel = label.map(x => math.log(x).toFloat)
    val rowkeyName = "Id"
    val labelName = "SalePrice"
    val dataPath = getClass.getClassLoader.getResource("house-prices-train.csv").getPath
    val testPath = getClass.getClassLoader.getResource("house-prices-test.csv").getPath
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val testSources = Source.fromFile(testPath, "utf-8").getLines()
    val (trainFeatures, testFeatures, trainLabels) = TmpUtils.preprocessing(sources, testSources, rowkeyName, labelName)
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    FLContext.initFLContext()
    val fGBoostRegression = new FGBoostRegression(
      learningRate = 0.1f, maxDepth = 7, minChildSize = 5)
    fGBoostRegression.fit(trainFeatures, trainLabels, 100)
    val fGBoostResult = fGBoostRegression.predict(testFeatures).map(tensor => tensor.value())
      .map(math.exp(_))
    // The predict result validation
    var cnt = 0
    fGBoostResult.indices.foreach(i => {
      val diffAllow = math.min(fGBoostResult(i), xGBoostResults(i)) * 0.05
      if (math.abs(fGBoostResult(i) - xGBoostResults(i)) < diffAllow) cnt += 1
    })
    flServer.stop()
    val fgBoostTreeInFormat = fGBoostRegression.trees.toArray.map(
      fTree => XGBoostFormatSerializer(fTree))
    // The tree structure validation
    XGBoostFormatValidator(fgBoostTreeInFormat, xgBoostFormatNodes)
    logger.info(s"Got similar result: ${cnt}/${fGBoostResult.length}")
    require(cnt > 900, s"Should get over 900 results similar with XGBoost, but got only: $cnt")
  }

  /**
   * A two client test for FGBoost
   * half of features are stored in one csv and another half in another csv
   * the label column is at xxx-csv-2 file
   */
  "FGBoost Correctness two parties" should "work" in {
    val rowkeyName = "Id"
    val labelName = "SalePrice"
    val dataPath = getClass.getClassLoader.getResource("house-prices-train.csv-1.csv").getPath
    val testPath = getClass.getClassLoader.getResource("house-prices-test.csv-1.csv").getPath
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val testSources = Source.fromFile(testPath, "utf-8").getLines()
    val (trainFeatures, testFeatures, trainLabels) =
      TmpUtils.preprocessing(sources, testSources, rowkeyName, labelName)

    val flServer = new FLServer()
    flServer.setClientNum(2)
    flServer.build()
    flServer.start()
    FLContext.initFLContext()
    val mockClient = new MockClient(
      dataPath = getClass.getClassLoader.getResource("house-prices-train.csv-2.csv").getPath,
      testPath = getClass.getClassLoader.getResource("house-prices-test.csv-2.csv").getPath,
      rowKeyName = "Id", labelName = "SalePrice", dataFormat = "raw")
    mockClient.start()
    val fGBoostRegression = new FGBoostRegression(
      learningRate = 0.1f, maxDepth = 7, minChildSize = 5)
    logger.debug(s"Client1 calling fit...")
    fGBoostRegression.fit(trainFeatures, trainLabels, 100)
    val fGBoostResult = fGBoostRegression.predict(testFeatures).map(tensor => tensor.value())
      .map(math.exp(_))
    var cnt = 0
    fGBoostResult.indices.foreach(i => {
      val diffAllow = math.min(fGBoostResult(i), xGBoostResults(i)) * 0.05
      if (math.abs(fGBoostResult(i) - xGBoostResults(i)) < diffAllow) cnt += 1
    })
    flServer.stop()
    logger.info(s"Got similar result: ${cnt}/${fGBoostResult.length}")
    require(cnt > 900, s"Should get over 900 results similar with XGBoost, but got only: $cnt")
//     uncomment following code if want to save to file (to make a submission maybe)
//    val file = new File("filePath")
//    val bw = new BufferedWriter(new FileWriter(file))
//    bw.write("Id,SalePrice\n")
//    val testOriginWithHeader = Source.fromFile(testPath, "utf-8").getLines().map(_.split(",").map(_.trim)).toArray
//    val testOrigin = testOriginWithHeader.slice(1, testOriginWithHeader.length)
//    xGBoostResults.zip(testOrigin).foreach{r =>
//      bw.write(s"${r._2(0)},${r._1}\n")
//    }
//    bw.close()
  }
}
