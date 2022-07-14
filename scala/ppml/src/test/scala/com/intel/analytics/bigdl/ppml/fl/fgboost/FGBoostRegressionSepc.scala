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

package com.intel.analytics.bigdl.ppml.fl.fgboost

import com.intel.analytics.bigdl.ppml.fl.{FLContext, FLServer, FLSpec}
import com.intel.analytics.bigdl.ppml.fl.algorithms.FGBoostRegression
import com.intel.analytics.bigdl.ppml.fl.data.PreprocessUtil
import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.XGBoostFormatValidator
import org.apache.log4j.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import java.util.UUID
import scala.io.Source

class FGBoostRegressionSepc extends FLSpec {
  // House pricing dataset compared with xgboost training and prediction result
  // TODO: use DataFrame API to do the same validation
  "FGBoostRegression save and load" should "work" in {
    val flServer = new FLServer()
    try {
      flServer.setPort(port)
      val rowkeyName = "Id"
      val labelName = "SalePrice"
      val dataPath = getClass.getClassLoader.getResource("house-prices-train.csv").getPath
      val testPath = getClass.getClassLoader.getResource("house-prices-test.csv").getPath
      val sources = Source.fromFile(dataPath, "utf-8").getLines()
      val testSources = Source.fromFile(testPath, "utf-8").getLines()
      val (trainFeatures, testFeatures, trainLabels, flattenHeaders) =
        PreprocessUtil.preprocessing(sources, testSources, rowkeyName, labelName)
      XGBoostFormatValidator.clearHeaders()
      XGBoostFormatValidator.addHeaders(flattenHeaders)
      flServer.build()
      flServer.start()
      FLContext.initFLContext("1", target)
      val fGBoostRegression = new FGBoostRegression(
        learningRate = 0.1f, maxDepth = 7, minChildSize = 5)
      fGBoostRegression.fit(trainFeatures, trainLabels, 1)
      val tmpFileName = s"/tmp/${UUID.randomUUID().toString}"
      fGBoostRegression.saveModel(tmpFileName)

      val fGBoostRegressionLoaded = FGBoostRegression.loadModel(tmpFileName)


      var cnt = 0
    } catch {
      case e: Exception => throw e
    } finally {
      flServer.stop()
    }

  }
}

