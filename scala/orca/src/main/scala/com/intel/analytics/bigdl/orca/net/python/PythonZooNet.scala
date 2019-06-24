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

import java.nio.ByteOrder
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, Trigger, ValidationMethod}
import com.intel.analytics.bigdl.python.api.{JTensor, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.net._
import com.intel.analytics.bigdl.dataset.{Sample => JSample}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.io.Source
import scala.reflect.ClassTag
import scala.reflect.io.Path
import scala.collection.mutable.ListBuffer
import java.util.ArrayList
import java.util.concurrent.{CopyOnWriteArrayList, TimeUnit}

import org.apache.log4j.{Level, Logger}

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

  def netLoadTF(path: String, inputs: JList[String], outputs: JList[String],
      byteOrder: String, binFile: String = null): AbstractModule[Activity, Activity, T] = {
    val order = byteOrder match {
      case "little_endian" => ByteOrder.LITTLE_ENDIAN
      case "big_endian" => ByteOrder.BIG_ENDIAN
      case _ => throw new IllegalArgumentException(s"No support byte order $byteOrder")
    }
    Net.loadTF[T](path, inputs.asScala, outputs.asScala, order, Option(binFile))
  }

  def netLoadTF(folder: String): AbstractModule[Activity, Activity, T] = {
    Net.loadTF[T](folder)
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

  def createTFNet(path: String): TFNet = {
    TFNet(path)
  }

  def createTFTrainingHelper(modelPath: String, config: Array[Byte] = null): TFTrainingHelper = {
    TFTrainingHelper(modelPath, config)
  }

  def createIdentityCriterion(): IdentityCriterion = {
    new IdentityCriterion()
  }

  def createMergeFeatureLabelImagePreprocessing(): MergeFeatureLabel = {
    new MergeFeatureLabel()
  }

  def createMergeFeatureLabelFeatureTransformer(): MergeFeatureLabel = {
    new MergeFeatureLabel()
  }

  def createTFValidationMethod(validationMethod: ValidationMethod[Float],
                               outputLength: Int, targetLength: Int): TFValidationMethod = {
    new TFValidationMethod(validationMethod, outputLength, targetLength)
  }

  def createTFOptimizer(modelPath: String,
                        optimMethod: OptimMethod[Float],
                        x: JavaRDD[Sample],
                        batchSize: Int = 32): TFOptimizer = {
    new TFOptimizer(modelPath, optimMethod,
      toJSample(x).asInstanceOf[RDD[JSample[Float]]], batchSize)
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

  def createTorchNet(path: String): TorchNet = {
    TorchNet(path)
  }

}
