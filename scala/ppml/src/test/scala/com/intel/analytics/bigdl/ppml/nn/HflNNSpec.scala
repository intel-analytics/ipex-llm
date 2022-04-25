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

package com.intel.analytics.bigdl.ppml.nn

import com.intel.analytics.bigdl.ppml.algorithms.hfl.LogisticRegression
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.{FLContext, FLServer}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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
    val lr = new LogisticRegression(df.columns.size - 1)
    lr.fit(df, valData = df)
  }
}
class NNSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger {
  "Logistic Regression" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    val spark = FLContext.getSparkSession()
    val trainDf = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val testDf = trainDf.drop("Outcome")
    trainDf.show()
    FLContext.initFLContext()
    val lr = new LogisticRegression(trainDf.columns.size - 1)
    lr.fit(trainDf, valData = trainDf)
    lr.evaluate(trainDf)
    lr.predict(testDf)
    flServer.stop()
  }
  "Linear Regression" should "work" in {

  }

}
