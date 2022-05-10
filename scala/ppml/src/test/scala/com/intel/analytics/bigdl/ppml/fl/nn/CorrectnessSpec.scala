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

import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fl.nn.MockClient
import com.intel.analytics.bigdl.ppml.fl.{FLContext, FLServer}
import org.apache.log4j.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import java.io.File
import scala.io.Source
import com.intel.analytics.bigdl.ppml.fl.utils.TestUtils

import scala.concurrent.Future

class CorrectnessSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger {
  // This is a full-dataset validation so we disable debug log
  val logger = LogManager.getLogger(getClass)

  "NN Correctness two parties" should "work" in {
    val flServer = new FLServer()
    try {
      val dataPath1 = getClass.getClassLoader.getResource("two-party/diabetes-vfl-1.csv").getPath
      val dataPath2 = getClass.getClassLoader.getResource("two-party/diabetes-vfl-2.csv").getPath

      flServer.setClientNum(2)
      flServer.build()
      flServer.start()
      val mockClient1 = new MockClient(
        dataPath = dataPath1,
        featureColumns = Array("Pregnancies","Glucose","BloodPressure","SkinThickness"),
        labelColumns = Array("Outcome")
      )
      val mockClient2 = new MockClient(
        dataPath = dataPath2,
        featureColumns = Array("Insulin","BMI","DiabetesPedigreeFunction"),
        labelColumns = null
      )
      @volatile var errorFlag = false
      val exceptionHandler = new Thread.UncaughtExceptionHandler {
        override def uncaughtException(thread: Thread, throwable: Throwable): Unit = {
          errorFlag = true
          throwable.printStackTrace()
        }
      }
      mockClient1.setUncaughtExceptionHandler(exceptionHandler)
      mockClient2.setUncaughtExceptionHandler(exceptionHandler)
      mockClient1.start()
      mockClient2.start()
      mockClient1.join()
      mockClient2.join()
      if (errorFlag) {
        throw new Exception("Test failed, check the log")
      }
    } catch {
      case e: Exception => throw e
    } finally {
      flServer.stop()
    }
  }
}

