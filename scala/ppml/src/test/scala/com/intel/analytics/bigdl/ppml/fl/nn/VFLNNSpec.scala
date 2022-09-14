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

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.fl.algorithms.{PSI, VFLLinearRegression, VFLLogisticRegression}
import com.intel.analytics.bigdl.ppml.fl.{FLContext, FLServer, FLSpec}


class VFLNNSpec extends FLSpec {
  "Logistic Regression DataFrame API" should "work" in {
    val flServer = new FLServer()
    flServer.setPort(port)
    flServer.build()
    flServer.start()
    val spark = FLContext.getSparkSession()
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)

    FLContext.initFLContext(1, target)
    val psi = new PSI()
    val salt = psi.getSalt()
    val trainDf = psi.uploadSetAndDownloadIntersectionDataFrame(df, salt)
    val testDf = trainDf.drop("Outcome")
    trainDf.show()
    val lr = new VFLLogisticRegression(df.columns.size - 1)
    lr.fitDataFrame(trainDf, valData = trainDf)
    lr.evaluateDataFrame(trainDf)
    lr.predictDataFrame(testDf)
    flServer.stop()
  }
  "Logistic Regression Tensor API" should "work" in {
    val flServer = new FLServer()
    flServer.setPort(port)
    flServer.build()
    flServer.start()
    FLContext.initFLContext(1, target)
    val xTrain = Tensor[Float](10, 10)
    val yTrain = Tensor[Float](10, 1)
    val lr = new VFLLogisticRegression(10)
    lr.fit(xTrain, yTrain)
    lr.evaluate(xTrain)
    lr.predict(xTrain)
    flServer.stop()
  }
  "Linear Regression Tensor API" should "work" in {
    val flServer = new FLServer()
    flServer.setPort(port)
    flServer.build()
    flServer.start()
    FLContext.initFLContext(1, target)
    val xTrain = Tensor[Float](10, 10)
    val yTrain = Tensor[Float](10, 1)
    val lr = new VFLLinearRegression(10)
    lr.fit(xTrain, yTrain)
    lr.evaluate(xTrain)
    lr.predict(xTrain)
    flServer.stop()
  }
  "Multiple algorithm" should "work" in {
    val flServer = new FLServer()
    flServer.setPort(port)
    flServer.build()
    flServer.start()
    FLContext.initFLContext(1, target)
    val logisticRegression = new VFLLogisticRegression(featureNum = 1)
    val linearRegression = new VFLLinearRegression(featureNum = 1)
    flServer.stop()
  }
}
