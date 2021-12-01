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

import com.intel.analytics.bigdl.ppml.FLServer
import com.intel.analytics.bigdl.ppml.algorithms.PSI
import com.intel.analytics.bigdl.ppml.algorithms.vfl.{LinearRegression, LogisticRegression}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.JavaConverters._


class NNSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "Logistic Regression" should "work" in {
    // TODO: tests would be added after API changed to Spark local DataFrame
  }
  "Linear Regression" should "work" in {

  }
  "Multiple algorithm" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    VflContext.initContext()
    val logisticRegression = new LogisticRegression(featureNum = 1)
    val linearRegression = new LinearRegression(featureNum = 1)
    flServer.stop()
  }
}
