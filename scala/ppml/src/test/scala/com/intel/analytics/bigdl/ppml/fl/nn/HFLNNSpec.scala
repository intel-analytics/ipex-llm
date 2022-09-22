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

import com.intel.analytics.bigdl.ppml.fl.algorithms.HFLLogisticRegression
import com.intel.analytics.bigdl.ppml.fl.{FLContext, FLServer, FLSpec}

class MockAnotherParty(algorithm: String, clientID: String = "mock") extends Thread {

  override def run(): Unit = {
    algorithm match {
      case "logistic_regression" => runLogisticRegression()
      case _ => throw new NotImplementedError()
    }
  }
  def runLogisticRegression(): Unit = {
    val spark = FLContext.getSparkSession()
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val lr = new HFLLogisticRegression(df.columns.size - 1)
    lr.fitDataFrame(df, valData = df)
  }
}
class NNSpec extends FLSpec {
  "Logistic Regression" should "work" in {
    val flServer = new FLServer()
    flServer.setPort(port)
    flServer.build()
    flServer.start()
    val spark = FLContext.getSparkSession()
    val trainDf = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val testDf = trainDf.drop("Outcome")
    trainDf.show()
    FLContext.initFLContext(1, target)
    val lr = new HFLLogisticRegression(trainDf.columns.size - 1)
    lr.fitDataFrame(trainDf, valData = trainDf)
    lr.evaluateDataFrame(trainDf)
    lr.predictDataFrame(testDf)
    flServer.stop()
  }
  "Linear Regression" should "work" in {

  }

}
