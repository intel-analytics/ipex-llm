/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.torch

import java.io._

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.TorchObject._
import com.intel.analytics.sparkdl.utils.{File, Table}

import scala.io.Source
import scala.sys.process._

/**
 * With parameters input, run method would automatically run on torch and return the
 * corresponding output.
 */
object TH {
  def hasTorch(): Boolean = {
    val torchPath = System.getProperty("torch_location")
    val exitValue = if (torchPath != null) s"ls $torchPath".! else "which th".!
    return exitValue == 0
  }

  // Run with map
  def run(code: String, parameters: Map[String, Any],
    result: Array[String]): (Double, Map[String, Any]) = {
    val suffix = ".t7"
    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)
    var resultMap: Map[String, Any] = Map()

    val luaTime = runNM(code: String, parameters: Map[String, Any], result: Array[String])

    result.foreach { k =>
      val tmp: Any = File.loadTorch(subPath + k + suffix)
      resultMap += (k -> tmp)
    }

    (luaTime, resultMap)
  }

  def run(path: java.nio.file.Path, parameters: Map[String, Tensor[Double]],
    result: Array[String]): (Double, Map[String, Any]) = {
    val code = new StringBuilder("")
    Source.fromFile(path.toString()).foreach { k =>
      code.append(k)
    }
    run(code.toString(), parameters, result)
  }

  // Run without map
  def runNM(code: String, parameters: Map[String, Any], result: Array[String]): Double = {
    val suffix = ".t7"
    val varCode = new StringBuilder("require 'nn'\n" + "require 'optim'\n")
    val usrCode = new StringBuilder("")
    val resCode = new StringBuilder("")

    // Variable load code of lua
    parameters.keys.foreach { k =>
      val tmp = java.io.File.createTempFile(k + "Tmp", suffix)
      val tmpPath = tmp.getAbsolutePath
      parameters(k) match {
        case _: Tensor[_] =>
          if (parameters(k).asInstanceOf[Tensor[_]].getType() == FloatType) {
            File.saveTorch(parameters(k), tmpPath, TYPE_FLOAT_TENSOR)
          } else {
            File.saveTorch(parameters(k), tmpPath, TYPE_DOUBLE_TENSOR)
          }
        case _: Linear[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_LINEAR)
        case _: SpatialConvolution[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_SPATIALCONVOLUTION)
        case _: SpatialMaxPooling[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_SPATIALMAXPOOLING)
        case _: Threshold[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_THRESHOLD)
        case _: Concat[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_CONCAT)
        case _: Sequential[_, _, _] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_SEQUENTIAL)
        case _: View[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_VIEW)
        case _: Dropout[_] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_DROPOUT)
        case _: Table =>
          File.saveTorch(parameters(k).asInstanceOf[Table].getState(), tmpPath, TYPE_TABLE)
        case _ =>
      }
      varCode.append(k + " = torch.load(\'" + tmpPath + "\')\n")
    }

    // Read from user`s code
    usrCode.append("Timer = torch.Timer()\n")
    usrCode.append(code)
    usrCode.append("\nluaTime = Timer:time().real\nprint(luaTime)")

    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)

    // Result save code of lua
    result.foreach { k =>
      resCode.append("torch.save(\'" + subPath + k + suffix + "\', " + k + ")\n")
    }
    val writer = new PrintWriter(tmpFile)
    println("\n============== lua code start ==============\n")
    println(varCode)
    println(usrCode + "\n")
    println(resCode)
    println("============== lua code end ================\n")
    writer.write(varCode.toString() + "\n\n")
    writer.write(usrCode.toString() + "\n\n")
    writer.write(resCode.toString() + "\n\n")
    writer.close()

    println(tmpFile.getAbsolutePath)

    var luaTime = Seq(System.getProperty("torch_location", "th"), tmpFile.getAbsolutePath).!!.trim

    println("luaTime:" + luaTime)

    var pattern = java.util.regex.Pattern.compile("[hms]")
    var matcher = pattern.matcher(luaTime)

    if (matcher.find()) {
      luaTime = luaTime.substring(matcher.start() + 1)
      pattern = java.util.regex.Pattern.compile("m")
      matcher = pattern.matcher(luaTime)

      if (matcher.find()) {
        luaTime = luaTime.substring(0, luaTime.length - 5)
      }
    } else {
      pattern = java.util.regex.Pattern.compile("m")
      matcher = pattern.matcher(luaTime)
      if (matcher.find()) {
        luaTime = luaTime.substring(0, luaTime.length - 5)
      }
    }

    luaTime.toDouble
  }

  // Single map
  def map(result: String): (Any) = {
    val suffix = ".t7"
    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)
    val tmp: Any = File.loadTorch(subPath + result + suffix)
    tmp
  }

}
