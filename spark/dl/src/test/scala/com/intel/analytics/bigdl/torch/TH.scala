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

package com.intel.analytics.bigdl.torch

import java.io._
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.TorchObject._
import com.intel.analytics.bigdl.utils.{File, Table}

import scala.io.Source
import scala.sys.process._

/**
 * With parameters input, run method would automatically run on torch and return the
 * corresponding output.
 */
class TH {
  private val uuid: String = shortUUID()
  private var savedParameters: Option[Map[String, Any]] = None
  private var savedResult: Option[Array[String]] = None
  private val torch_location = System.getProperty("torch_location")
  private val torch = if (torch_location == null) {
    "th"
  } else {
    Paths.get(torch_location, "th").toString
  }

  def hasTorch(): Boolean = {
    val torchPath = System.getProperty("torch_location")
    // Skip on windows
    if (System.getProperty("os.name").toLowerCase().contains("win")) {
      return false
    }
    val exitValue = if (torchPath != null) s"ls $torchPath".! else "which th".!
    exitValue == 0
  }

  // Run with map
  def run(code: String, parameters: Map[String, Any],
    result: Array[String]): (Double, Map[String, Any]) = {
    var resultMap: Map[String, Any] = Map()

    val luaTime = runNM(code: String, parameters: Map[String, Any], result: Array[String])

    result.foreach { k =>
      val name = tmpFileName(k)
      val tmp: Any = File.loadTorch(name)
      resultMap += (k -> tmp)
    }

    (luaTime, resultMap)
  }

  def run(path: java.nio.file.Path, parameters: Map[String, Tensor[Double]],
    result: Array[String]): (Double, Map[String, Any]) = {
    val code = new StringBuilder("")
    Source.fromFile(path.toString).foreach { k =>
      code.append(k)
    }
    run(code.toString(), parameters, result)
  }

  // Run without map
  def runNM(code: String, parameters: Map[String, Any], result: Array[String]): Double = {
    val varCode = new StringBuilder("require 'nn'\n" + "require 'optim'\n")
    val usrCode = new StringBuilder("")
    val resCode = new StringBuilder("")

    if (savedParameters.isDefined) {
      release()
    } else {
      savedParameters = Some(parameters)
      savedResult = Some(result)
    }

    // Variable load code of lua
    parameters.keys.foreach { k =>
      val tmpPath = tmpFileName(k, "Tmp")
      parameters(k) match {
        case _: Tensor[_] =>
          if (parameters(k).asInstanceOf[Tensor[_]].getType() == FloatType) {
            File.saveTorch(parameters(k), tmpPath, TYPE_FLOAT_TENSOR, true)
          } else {
            File.saveTorch(parameters(k), tmpPath, TYPE_DOUBLE_TENSOR, true)
          }
        case _: AbstractModule[_, _, _] =>
          File.saveTorch(parameters(k), tmpPath, TYPE_MODULE, true)
        case _: Table =>
          File.saveTorch(parameters(k).asInstanceOf[Table], tmpPath, TYPE_TABLE, true)
        case _ =>
      }
      varCode.append(k + " = torch.load(\'" + tmpPath + "\')\n")
    }

    // Read from user`s code
    usrCode.append("Timer = torch.Time()\n")
    usrCode.append(code)
    usrCode.append("\nluaTime = Timer:time().real\nprint(luaTime)")

    val tmpFile = tmpFileName("torch", suffix = "lua")

    // Result save code of lua
    result.foreach { k =>
      resCode.append("torch.save(\'" + tmpFileName(k) + "\', " + k + ")\n")
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

    println(tmpFile)

    val stdout = new StringBuilder
    val stderr = new StringBuilder
    val status = s"$torch $tmpFile" ! ProcessLogger(stdout append _, stderr append _)

    var luaTime = if (status == 0) { // successful
      stdout.toString.trim
    } else {
      throw new RuntimeException(s"Nonzero exit value: $status ${stderr
              .replaceAllLiterally("\t", "\n")}")
    }

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
    val tmpFile = tmpFileName(result)
    val tmp: Any = File.loadTorch(tmpFile)
    tmp
  }

  def release(result: Map[String, Any]): Unit = {
    result.foreach { k =>
      val name = tmpFileName(k._1)
      Files.deleteIfExists(Paths.get(name))
    }

    Files.deleteIfExists(Paths.get(tmpFileName("torch", suffix = "lua")))

    if (savedParameters != null) {
      savedParameters.get.foreach { k =>
        val name = tmpFileName(k._1, infix = "Tmp")
        Files.deleteIfExists(Paths.get(name))
      }
    }
  }

  def release(result: Array[String]): Unit = {
    result.foreach { k =>
      val name = tmpFileName(k)
      Files.deleteIfExists(Paths.get(name))
    }

    Files.deleteIfExists(Paths.get(tmpFileName("torch", suffix = "lua")))

    savedParameters.get.foreach { k =>
      val name = tmpFileName(k._1, infix = "Tmp")
      Files.deleteIfExists(Paths.get(name))
    }
  }

  def release(): Unit = {
    release(savedResult.get)
    savedResult = null
    savedParameters = null
  }

  def shortUUID(): String = {
    val uuid = UUID.randomUUID()
    val l = ByteBuffer.wrap(uuid.toString.getBytes()).getLong()
    java.lang.Long.toString(l, Character.MAX_RADIX)
  }

  private def tmpFileName(name: String, infix: String = "", suffix: String = "t7"): String = {
    val fileName = List(name + infix, uuid, suffix).mkString(".")
    val tmpDir = System.getProperty("java.io.tmpdir")
    Paths.get(tmpDir, fileName).toAbsolutePath.toString
  }

}

