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

package com.intel.analytics.bigdl.ppml.fl.xgboost

import com.intel.analytics.bigdl.ppml.fl.data.PreprocessUtil
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.{XGBoostFormatSerializer, XGBoostFormatValidator}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}

import java.io.{BufferedWriter, File, FileWriter}
import scala.io.Source

/**
 * This is an XGBoost pipeline running House Price dataset with 15 boosting round.
 * For now FGBoost validation use the first 15 boosting tree to validate if they are
 * consistent with XGBoost trees.
 * Result should not be considered with the training labels because only 15 rounds are boosted.
 */
object XGBoostPipeline {
  def process(dataPathTrain: String,
              dataPathTest: String,
              predictResultPath: String,
              treeDumpPath: String): Unit = {
    val sources = Source.fromFile(dataPathTrain, "utf-8").getLines()
    val testSources = Source.fromFile(dataPathTest, "utf-8").getLines()
    val rowkeyName = "Id"
    val labelName = "SalePrice"
    val (trainFeatures, testFeatures, trainLabels, flattenHeaders) = {
      PreprocessUtil.preprocessing(sources, testSources, rowkeyName, labelName)
    }

    XGBoostFormatValidator.setXGBoostHeaders(flattenHeaders)
    val trainFeatureArray = trainFeatures.map(tensor => tensor.toArray()).flatten
    val testFeatureArray = testFeatures.map(tensor => tensor.toArray()).flatten
    val trainDMat = new DMatrix(trainFeatureArray, trainFeatures.length,
      trainFeatureArray.length / trainFeatures.length, 0.0f)
    val testDMat = new DMatrix(testFeatureArray, testFeatures.length,
      testFeatureArray.length / testFeatures.length, 0.0f)
    trainDMat.setLabel(trainLabels)
    val params = Map("eta" -> 0.1, "max_depth" -> 7, "objective" -> "reg:squarederror",
      "min_child_weight" -> 5)
    val booster = XGBoost.train(trainDMat, params, 15)
    val xgBoostResults = booster.predict(testDMat).flatten.map(math.exp(_))
    val treeStr = booster.getModelDump(featureMap = null, withStats = false, format = "json")
    // XGBoost model dump will get 100(number of boost round) trees, we validate each one

    val resultWriter = new BufferedWriter(new FileWriter(new File(predictResultPath)))
    resultWriter.write(s"Id,SalePrice\n")
    xgBoostResults.indices.foreach(i => resultWriter.write(s"${i + 1},${xgBoostResults(i)}\n"))
    resultWriter.close()

    val treeDumpWriter = new BufferedWriter(new FileWriter(new File(treeDumpPath)))
    treeDumpWriter.write(treeStr.mkString("\n\n\n"))
    treeDumpWriter.close()
  }

  def main(args: Array[String]): Unit = {
    process(getClass.getClassLoader.getResource("house-prices-train.csv").getPath,
    getClass.getClassLoader.getResource("house-prices-test.csv").getPath,
    "house-price-xgboost-submission.csv",
    "house-price-xgboost-tree-dump")
  }
}
