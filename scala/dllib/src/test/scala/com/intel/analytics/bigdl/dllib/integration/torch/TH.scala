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

package com.intel.analytics.bigdl.integration.torch

import java.io._
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.TorchObject._
import com.intel.analytics.bigdl.utils.{File, Table}
import org.apache.commons.lang.SerializationUtils

import scala.collection.immutable.ListMap
import scala.io.Source
import scala.sys.process._

/**
 * With parameters input, run method would automatically run on torch and return the
 * corresponding output.
 */
object TH {
  def hasTorch(): Boolean = {
    val torchPath = System.getProperty("torch_location")
    // Skip on windows
    if (System.getProperty("os.name").toLowerCase().contains("win")) {
      return false
    }
    val exitValue = if (torchPath != null) s"ls $torchPath".! else "which th".!
    return exitValue == 0
  }

  def hasRNN: Boolean = {
    val tmpFile = java.io.File.createTempFile("checkRNN", ".lua", scriptsRoot.toFile)
    val writer = new PrintWriter(tmpFile)
    writer.write("exist = (pcall(require, 'rnn'))\n print(exist)")
    writer.close()

    val existsRNN =
      Seq(System.getProperty("torch_location", "th"), tmpFile.getAbsolutePath).!!.trim

    existsRNN.contains("true")
  }

  private def getRoot(subDir: String): Path = {
    val tmpDir = System.getProperty("java.io.tmpdir")
    val root = Paths.get(tmpDir, subDir)
    if (Files.notExists(root)) {
      Files.createDirectory(root)
    }
    root
  }

  val resultsRoot: Path = {
    getRoot("torch-results")
  }

  val scriptsRoot: Path = {
    getRoot("torch-scripts")
  }

  val inputsRoot: Path = {
    getRoot("torch-inputs")
  }

  val timeSuffix: String = ".time"

  // Run with map
  def run(code: String, parameters: Map[String, Any],
    result: Array[String])(implicit id: TestCaseIdentity): (Double, Map[String, Any]) = {
    val suffix = id.suffix
    var resultMap: Map[String, Any] = Map()

    val luaTime = if (isExists(result, id.suffix)) {
      val path = Paths.get(resultsRoot.toAbsolutePath.toString, suffix + timeSuffix).toString
      File.load[Array[Double]](path).head
    } else {
      runNM(code: String, parameters: Map[String, Any], result: Array[String], suffix)
    }

    result.foreach { k =>
      val subPath = Paths.get(resultsRoot.toAbsolutePath.toString, k + suffix).toString
      val tmp: Any = File.loadTorch(subPath)
      resultMap += (k -> tmp)
    }

    (luaTime, resultMap)
  }

  def run(path: java.nio.file.Path, parameters: Map[String, Tensor[Double]],
    result: Array[String])(implicit id: TestCaseIdentity): (Double, Map[String, Any]) = {
    val code = new StringBuilder("")
    Source.fromFile(path.toString()).foreach { k =>
      code.append(k)
    }
    run(code.toString(), parameters, result)
  }

  // Run without map
  def runNM(code: String, parameters: Map[String, Any], result: Array[String], suffix: String)
  : Double = {
    if (isExists(result, suffix)) {
      val luaTime = {
        val path = Paths.get(resultsRoot.toString, suffix + timeSuffix)
        File.load[Array[Double]](path.toString).head
      }

      return luaTime // stop early
    }

    val varCode = new StringBuilder("require 'nn'\n" + "require 'optim'\n")
    val usrCode = new StringBuilder("")
    val resCode = new StringBuilder("")

    // Variable load code of lua
    parameters.keys.foreach { k =>
      // sometimes the k is too short, createTempFile will failed.
      // so we just need to swap the k and suffix
      val tmp = try {
        java.io.File.createTempFile(k, suffix, inputsRoot.toFile)
      } catch {
        case illegalArgumentException: IllegalArgumentException =>
          java.io.File.createTempFile(suffix, k, inputsRoot.toFile)
        case iOException: IOException => throw iOException
      }

      val inputsPath = tmp.getAbsolutePath
      parameters(k) match {
        case _: Tensor[_] =>
          if (parameters(k).asInstanceOf[Tensor[_]].getType() == FloatType) {
            File.saveTorch(parameters(k), inputsPath, TYPE_FLOAT_TENSOR, true)
          } else {
            File.saveTorch(parameters(k), inputsPath, TYPE_DOUBLE_TENSOR, true)
          }
        case _: AbstractModule[_, _, _] =>
          File.saveTorch(parameters(k), inputsPath, TYPE_MODULE, true)
        case _: Table =>
          File.saveTorch(parameters(k).asInstanceOf[Table], inputsPath, TYPE_TABLE, true)
        case _ =>
      }
      varCode.append(k + " = torch.load(\'" + inputsPath + "\')\n")
    }

    // Read from user`s code
    usrCode.append("Timer = torch.Timer()\n")
    usrCode.append(code)
    usrCode.append("\nluaTime = Timer:time().real\nprint(luaTime)")

    val tmpFile = java.io.File.createTempFile("UnitTest", "lua", scriptsRoot.toFile)

    // Result save code of lua
    result.foreach { k =>
      val savePath = Paths.get(resultsRoot.toAbsolutePath.toString, k + suffix)
      resCode.append("torch.save(\'" + savePath + "\', " + k + ")\n")
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

    val timePath = Paths.get(resultsRoot.toAbsolutePath.toString, suffix + timeSuffix)
    File.save(Array[Double](luaTime.toDouble), timePath.toString)

    luaTime.toDouble
  }

  // Single map
  def map(result: String, suffix: String): (Any) = {
    val subPath = Paths.get(resultsRoot.toAbsolutePath.toString, result + suffix)
    val tmp: Any = File.loadTorch(subPath.toAbsolutePath.toString)
    tmp
  }

  def isExists(results: Array[String], suffix: String): Boolean = {
    val tensors = results.forall { result =>
      val path = Paths.get(resultsRoot.toAbsolutePath.toString, result + suffix)
      Files.exists(path)
    }
    val time = {
      val path = Paths.get(resultsRoot.toAbsolutePath.toString, suffix + timeSuffix)
      Files.exists(path)
    }

    tensors && time
  }
}
