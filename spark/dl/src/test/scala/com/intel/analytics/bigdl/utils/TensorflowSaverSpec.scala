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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.nn.{Graph, Linear, ReLU}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.sys.process._
import java.io.{File => JFile}

import com.intel.analytics.bigdl.utils.TestUtils.processPath

class TensorflowSaverSpec extends FlatSpec with Matchers {
  "ReLU layer" should "be correctly saved" in {
    val relu = ReLU().setName("relu").apply()
    val graph = Graph(relu, relu)

    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest", "ReLU")
    TensorFlowSaver.saveGraph(graph, Seq(("input", Seq(2, 4))), tmpFile.getPath)
    runPython(testScriptsPath("ReLUSaveTest.py ") + tmpFile) should be(true)
  }

  "Linear layer" should "be correctly saved" in {
    val linear = Linear(3, 4).setName("linear").apply()
    val graph = Graph(linear, linear)
    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest", "Linear")
    TensorFlowSaver.saveGraph(graph, Seq(("input", Seq(2, 3))), tmpFile.getPath)
    println(tmpFile.getPath)
    // runPython(testScriptsPath("LinearSaveTest.py ") + tmpFile) should be(true)
  }

  private def testScriptsPath(script: String) : String = {
    val resource = getClass().getClassLoader().getResource("tf")
    processPath(resource.getPath()) + JFile.separator + "saveTest" +
      JFile.separator + script
  }

  private def runPython(cmd: String): Boolean = {
    try {
      (("python " + cmd) !!)
      return true
    } catch {
      case _: Throwable => false
    }
  }
}