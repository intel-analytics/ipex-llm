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

import com.intel.analytics.bigdl.ppml.{FLContext, FLServer}
import com.intel.analytics.bigdl.ppml.algorithms.vfl.FGBoostRegression
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FGBoostRegressionSepc extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger{
  // House pricing dataset compared with xgboost training and prediction result
  "FGBoost Regression single party" should "work" in {
    val spark = FLContext.getSparkSession()
    val df = spark.read.option("header", "true")
      .csv(getClass.getClassLoader.getResource("house-prices-test.csv").getPath)
    val filledDF = DataFrameUtils.fillNA(df)
    filledDF.show()

    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    FLContext.initFLContext()
    val fGBoostRegression = new FGBoostRegression()
    fGBoostRegression.fit(filledDF)
    val result = fGBoostRegression.predict(df)
    require(result.size == 5, "output size mismatch")
  }
}

