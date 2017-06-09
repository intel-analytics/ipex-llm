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
package com.intel.analytics.bigdl.utils.tf

import com.intel.analytics.bigdl.utils.TestUtils.processPath
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import java.io.{File => JFile}

import org.apache.log4j.Logger

import scala.sys.process._

class TensorflowSpecHelper extends FlatSpec with Matchers with BeforeAndAfter {

  private val logger = Logger.getLogger(getClass)

  protected def tfCheck(): Unit = {
    var exitValue : String = ""
    try {
      exitValue = ((Seq("python", "-c", "import sys; print ','.join(sys.path)"))!!)
      ((Seq("python", "-c", "import tensorflow"))!!)
    } catch {
      case _: Throwable => cancel("python or tensorflow is not installed")
    }

    if (!exitValue.contains("models")) {
      cancel("Tensorflow models path is not exported")
    }
  }

  protected def runPython(cmd: String): Boolean = {
    try {
      logger.info("run command\n" + cmd)
      val proc = s"python $cmd".run
      return proc.exitValue() == 0
    } catch {
      case _: Throwable => false
    }
  }

  protected def runPythonSaveTest(graphPath: String, outputSuffix: String) : Boolean = {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      s"save_test.py $graphPath $outputSuffix"
    runPython(path)
  }
}
