/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.common

import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.net.NetUtils
import jep.{NDArray, SharedInterpreter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

@PythonInterpreterTest
class PythonInterpreterSpec extends ZooSpecHelper{
  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Logger.getLogger(PythonInterpreter.getClass()).setLevel(Level.DEBUG)
      // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
      TFNetNative.isLoaded
    }
  }

  "interp" should "work in all thread" in {
    ifskipTest()
    val code =
      s"""
         |import numpy as np
         |a = np.array([1, 2, 3])
         |""".stripMargin
    PythonInterpreter.exec(code)
    println(PythonInterpreter.getValue[NDArray[_]]("a").getData())
    (0 until 1).toParArray.foreach{i =>
      println(Thread.currentThread())
      PythonInterpreter.exec(code)
    }
    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("app").setMaster("local[4]"))
    (0 to 10).foreach(i =>
      sc.parallelize(0 to 10, 1).mapPartitions(i => {
        println(Thread.currentThread())
        PythonInterpreter.exec("a = np.array([1, 2, 3])")
        i
      }).count()
    )
    sc.stop()
  }
}
