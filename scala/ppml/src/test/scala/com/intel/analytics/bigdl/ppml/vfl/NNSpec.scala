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

package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.ppml.{FLContext, FLServer}
import com.intel.analytics.bigdl.ppml.algorithms.PSI
import com.intel.analytics.bigdl.ppml.algorithms.vfl.{LinearRegression, LogisticRegression}
import com.intel.analytics.bigdl.ppml.example.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.JavaConverters._

class MockAnotherParty(algorithm: String, clientID: String = "mock") extends Thread {
  override def run(): Unit = {
    algorithm match {
      case "logistic_regression" => runLogisticRegression()
      case _ => throw new NotImplementedError()
    }
  }
  def runLogisticRegression(): Unit = {
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val lr = new LogisticRegression(df.columns.size - 1)
    lr.fit(df, valData = df)
  }
}
class NNSpec extends FlatSpec with Matchers with BeforeAndAfter with LogManager {
  "Logistic Regression" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)

    FLContext.initFLContext()
    val psi = new PSI()
    val salt = psi.getSalt()
    val intersectionDf = psi.uploadSetAndDownloadIntersection(df, salt)
    val lr = new LogisticRegression(df.columns.size - 1)
    lr.fit(intersectionDf, valData = intersectionDf)
    lr.evaluate(intersectionDf)
    lr.predict(intersectionDf)
  }
  "Linear Regression" should "work" in {

  }
  "Multiple algorithm" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    FLContext.initFLContext()
    val logisticRegression = new LogisticRegression(featureNum = 1)
    val linearRegression = new LinearRegression(featureNum = 1)
    flServer.stop()
  }
}
