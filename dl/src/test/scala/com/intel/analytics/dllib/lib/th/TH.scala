package com.intel.analytics.dllib.lib.th

import java.io._

import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.tensor.TensorType.FloatType
import com.intel.analytics.dllib.lib.tensor.TorchObject._
import com.intel.analytics.dllib.lib.tensor._

import scala.io.Source
import scala.sys.process._

/**
  *With parameters input, run method would automatically run on torch and return the corresponding output.
  */


object TH {
  def hasTorch(): Boolean ={
    val exitValue = "which th".!
    return exitValue != 0 && System.getProperty("torch_location") == null
  }

  //run with map
  def run(code : String, parameters : Map[String, Any], result : Array[String]) : (Double, Map[String, Any]) = {
    val suffix = ".t7"
    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)
    var resultMap : Map[String, Any] = Map()

    val luaTime = runNM(code : String, parameters : Map[String, Any], result : Array[String])

    //map
    result.foreach{ k =>
      val tmp : Any = torch.load(subPath + k + suffix)
      resultMap += (k -> tmp)
    }

    (luaTime, resultMap)
  }

  def run(path : java.nio.file.Path, parameters : Map[String, Tensor[Double]], result : Array[String]) : (Double, Map[String, Any]) = {
    var code = new StringBuilder("")
    Source.fromFile(path.toString()).foreach { k =>
      code.append(k)
    }
    run(code.toString(), parameters, result)
  }

  //run without map
  def runNM(code : String, parameters : Map[String, Any], result : Array[String]): Double = {
    val suffix = ".t7"
    val varCode = new StringBuilder("require 'nn'\n" + "require 'optim'\n")
    val usrCode = new StringBuilder("")
    val resCode = new StringBuilder("")

    //variable load code of lua
    parameters.keys.foreach{ k=>
      val tmp = java.io.File.createTempFile(k + "Tmp", suffix)
      val tmpPath = tmp.getAbsolutePath
      //println("Temp file : " + tmpPath);
      parameters(k) match {
        case _: Tensor[Double] =>
          if(parameters(k).asInstanceOf[Tensor[_]].getType() == FloatType){
            torch.save(parameters(k), tmpPath, TYPE_FLOAT_TENSOR)
          }else {
            torch.save(parameters(k), tmpPath, TYPE_DOUBLE_TENSOR)
          }
        case _: Linear[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_LINEAR)
        case _: SpatialConvolution[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_SPATIALCONVOLUTION)
        case _: SpatialMaxPooling[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_SPATIALMAXPOOLING)
        case _: Threshold[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_THRESHOLD)
        case _: Concat[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_CONCAT)
        case _: Sequential[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_SEQUENTIAL)
        case _: View[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_VIEW)
        case _: Dropout[Double] =>
          torch.save(parameters(k), tmpPath, TYPE_DROPOUT)
        case _ =>
      }
      varCode.append(k + " = torch.load(\'" +  tmpPath + "\')\n")
    }

    //read from user`s code
    usrCode.append("Timer = torch.Timer()\n")
    usrCode.append(code)
    usrCode.append("\nluaTime = Timer:time().real\nprint(luaTime)")

    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)

    //result save code of lua

    result.foreach{ k =>
      resCode.append("torch.save(\'" + subPath + k + suffix + "\', " + k + ")\n")
    }
    val writer = new PrintWriter(tmpFile)
    println("\n============== lua code start ==============\n")
    println(varCode)
    println(usrCode+"\n")
    println(resCode)
    println("============== lua code end ================\n")
    writer.write(varCode.toString() + "\n\n")
    writer.write(usrCode.toString() + "\n\n")
    writer.write(resCode.toString() + "\n\n")
    writer.close()

    println(tmpFile.getAbsolutePath)

    var luaTime = Seq(System.getProperty("torch_location","th"), tmpFile.getAbsolutePath).!!.trim

    println("luaTime:" + luaTime)

    var pattern = java.util.regex.Pattern.compile("[hms]")
    var matcher = pattern.matcher(luaTime)

    if(matcher.find()) {
      luaTime = luaTime.substring(matcher.start()+1)
      pattern = java.util.regex.Pattern.compile("m")
      matcher = pattern.matcher(luaTime)

      if(matcher.find()) {
        luaTime = luaTime.substring(0, luaTime.length-5)
      }
    } else {
      pattern = java.util.regex.Pattern.compile("m")
      matcher = pattern.matcher(luaTime)
      if(matcher.find()) {
        luaTime = luaTime.substring(0, luaTime.length-5)
      }
    }

    luaTime.toDouble
  }

  //single map
  def map(result : String) : (Any) = {
    val suffix = ".t7"
    val tmpFile = java.io.File.createTempFile("UnitTest", "lua")
    val absolutePath = tmpFile.getAbsolutePath
    val subPath = absolutePath.substring(0, absolutePath.lastIndexOf(java.io.File.separator) + 1)
    val tmp : Any = torch.load(subPath + result + suffix)
    tmp
  }

}
