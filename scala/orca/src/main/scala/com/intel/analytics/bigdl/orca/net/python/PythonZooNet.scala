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
package com.intel.analytics.zoo.pipeline.api.net.python

import java.util.concurrent.{CopyOnWriteArrayList, TimeUnit}
import java.util.{ArrayList, List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.python.api.JTensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.net._
import org.apache.log4j.{Level, Logger}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag


object PythonZooNet {

  def ofFloat(): PythonZooNet[Float] = new PythonZooNet[Float]()

  def ofDouble(): PythonZooNet[Double] = new PythonZooNet[Double]()

}


class PythonZooNet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def newGraph(model: NetUtils[T, _],
               outputs: JList[String]): NetUtils[T, _] = {
    model.newGraph(outputs.asScala).asInstanceOf[NetUtils[T, _]]
  }

  def freezeUpTo(model: NetUtils[T, _], names: JList[String]): Unit = {
    model.freezeUpTo(names.asScala: _*)
  }

  def netLoadBigDL(
                    modulePath: String,
                    weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadBigDL[T](modulePath, weightPath)
  }

  def netLoadCaffe(
                    defPath: String,
                    modelPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadCaffe[T](defPath, modelPath)
  }

  def netLoad(
               modulePath: String,
               weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.load[T](modulePath, weightPath)
  }

  def netLoadTorch(
                    path: String): AbstractModule[Activity, Activity, T] = {
    Net.loadTorch[T](path)
  }

  def netToKeras(value: NetUtils[T, _]): KerasLayer[Activity, Activity, T] = {
    value.toKeras()
  }

  def createTFNet(
                   path: String,
                   inputNames: JList[String],
                   outputNames: JList[String]): TFNet = {
    TFNet(path, inputNames.asScala.toArray, outputNames.asScala.toArray)
  }

  def createTFNet(
                   path: String,
                   inputNames: JList[String],
                   outputNames: JList[String], config: Array[Byte]): TFNet = {
    TFNet(path, inputNames.asScala.toArray, outputNames.asScala.toArray, config)
  }

  def createTFNet(path: String): TFNet = {
    TFNet(path)
  }

  def createTFNetFromSavedModel(path: String,
                                tag: String,
                                inputNames: JList[String],
                                outputNames: JList[String],
                                config: Array[Byte],
                                initOp: String): Module[Float] = {
    val sessionConfig = Option(config).getOrElse(TFNet.defaultSessionConfig.toByteArray())
    TFNetForInference.fromSavedModel(path, Option(tag), None,
      Option(inputNames.asScala.toArray), Option(outputNames.asScala.toArray),
      sessionConfig, Option(initOp))
  }

  def createTFNetFromSavedModel(path: String,
                                tag: String,
                                signature: String,
                                config: Array[Byte]): Module[Float] = {
    if (config == null) {
      TFNet.fromSavedModel(path, tag, signature)
    } else {
      TFNetForInference.fromSavedModel(path, Option(tag), Option(signature),
        None, None, config)
    }
  }

  def createTFNet(path: String, config: Array[Byte]): TFNet = {
    TFNet(path, config)
  }

  val processToBeKill = new CopyOnWriteArrayList[String]()
  registerKiller()

  private def killPids(killingList: JList[String], killCommand: String): Unit = {
    try {
      val iter = killingList.iterator()
      while(iter.hasNext) {
        val pid = iter.next()
        println("JVM is stopping process: " +  pid)
        val process = Runtime.getRuntime().exec(killCommand + pid)
        process.waitFor(2, TimeUnit.SECONDS)
        if (process.exitValue() == 0) {
          iter.remove()
        }
      }
    } catch {
      case e : Exception =>
    }
  }

  private def registerKiller(): Unit = {
    Logger.getLogger("py4j.reflection.ReflectionEngine").setLevel(Level.ERROR)
    Logger.getLogger("py4j.GatewayConnection").setLevel(Level.ERROR)
    Runtime.getRuntime().addShutdownHook(new Thread {
          override def run(): Unit = {
            // Give it a chance to be gracefully killed
            killPids(processToBeKill, "kill ")
            if (!processToBeKill.isEmpty) {
              Thread.sleep(2000)
              killPids(processToBeKill, "kill -9")
            }
          }
      })
  }

  def jvmGuardRegisterPids(pids: ArrayList[Integer]): Unit = {
    pids.asScala.foreach(pid => processToBeKill.add(pid + ""))
  }

  def createTorchNet(modelPath: String): TorchNet = {
      TorchNet(modelPath)
  }

  def createTorchCriterion(lossPath: String): TorchCriterion = {
    TorchCriterion(lossPath)
  }

  def createTorchModel(model: Array[Byte], weights: JTensor): TorchModel = {
    TorchModel(model, weights.storage)
  }

  def createTorchLoss(criterion: Array[Byte]): TorchLoss = {
    TorchLoss(criterion)
  }

  def torchNetSavePytorch(torchnet: TorchNet, path: String): Unit = {
    torchnet.savePytorch(path)
  }

}
