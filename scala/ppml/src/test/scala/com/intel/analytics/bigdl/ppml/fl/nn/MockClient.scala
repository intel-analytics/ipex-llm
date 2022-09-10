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

package com.intel.analytics.bigdl.ppml.fl.nn

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.ppml.fl.FLContext
import com.intel.analytics.bigdl.ppml.fl.algorithms.{FGBoostRegression, VFLLogisticRegression}
import com.intel.analytics.bigdl.ppml.fl.utils.{FlContextForTest, TensorUtils}
import org.apache.log4j.LogManager

import scala.io.Source

class MockClient(clientId: Int,
                 dataPath: String,
                 featureColumns: Array[String] = null,
                 labelColumns: Array[String] = null,
                 learningRate: Float = 0.005f,
                 target: String = null) extends Thread {

  val logger = LogManager.getLogger(getClass)
  val testFlContext = new FlContextForTest()
  testFlContext.initFLContext(clientId, target)
  override def run(): Unit = {
    rawDataPipeline()
  }

  def rawDataPipeline(): Array[Activity] = {
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val spark = FLContext.getSparkSession()
    val dfTrain = spark.read.option("header", "true").csv(dataPath)
    val dfTest = dfTrain.drop("Outcome")
    val xTrain = TensorUtils.fromDataFrame(dfTrain, featureColumns)
    val yTrain = TensorUtils.fromDataFrame(dfTrain, labelColumns)
    val xTest = TensorUtils.fromDataFrame(dfTest, featureColumns)
    val lr = new VFLLogisticRegression(featureColumns.length, learningRate)
    lr.estimator.setFlClient(testFlContext.getClient())
    lr.fit(xTrain, yTrain, epoch = 5)
    lr.evaluate(xTrain, yTrain)
    lr.predict(xTest)
  }
}
