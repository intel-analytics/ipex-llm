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

package com.intel.analytics.bigdl.ppml.utils

import com.intel.analytics.bigdl.ppml.algorithms.vfl.FGBoostRegression
import org.apache.log4j.LogManager

import scala.io.Source

class MockClient(dataPath: String,
                 testPath: String = null,
                 rowKeyName: String = null,
                 labelName: String = null,
                 dataFormat: String = "raw") extends Thread {

  val logger = LogManager.getLogger(getClass)

  override def run(): Unit = {
    dataFormat match {
      case "raw" => rawDataPipeline()
      case _ => throw new IllegalArgumentException()
    }

  }
  def rawDataPipeline() = {
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val testSources = if (testPath != null) {
      Source.fromFile(testPath, "utf-8").getLines()
    } else null
    val (trainFeatures, testFeatures, trainLabels) =
      TmpUtils.preprocessing(sources, testSources, rowKeyName, labelName)
    val fgBoostRegression = new FGBoostRegression(
      learningRate = 0.1f, maxDepth = 7, minChildSize = 5)
    val testFlContext= new FlContextForTest()
    testFlContext.initFLContext()
    fgBoostRegression.flClient = testFlContext.getClient()
    logger.debug(s"Client2 calling fit...")
    fgBoostRegression.fit(trainFeatures, trainLabels, 100)
    fgBoostRegression.predict(testFeatures)
  }
}
