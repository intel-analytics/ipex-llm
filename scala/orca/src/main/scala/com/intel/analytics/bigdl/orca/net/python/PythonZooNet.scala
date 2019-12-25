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

import java.nio.FloatBuffer
import java.util.concurrent.{CopyOnWriteArrayList, TimeUnit}
import java.util.{ArrayList, List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{JTensor, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.{PythonZoo, RDDWrapper}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.net._
import com.intel.analytics.zoo.tfpark._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import org.tensorflow.{Graph, Session, Tensor => TTensor}

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
                                config: Array[Byte]): Module[Float] = {
    if (config == null) {
      TFNet.fromSavedModel(path, tag, inputNames.asScala.toArray, outputNames.asScala.toArray)
    } else {
      TFNetForInference.fromSavedModel(path, Option(tag), None,
        Option(inputNames.asScala.toArray), Option(outputNames.asScala.toArray), config)
    }
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

  def createTFTrainingHelper(modelPath: String, config: Array[Byte] = null): Module[Float] = {
    TFTrainingHelper(modelPath, config)
  }

  def saveCheckpoint(model: TFTrainingHelper): Unit = {
    model.saveCheckpoint()
  }

  def createIdentityCriterion(): IdentityCriterion = {
    new IdentityCriterion()
  }

  def createMergeFeatureLabelImagePreprocessing(): MergeFeatureLabel = {
    new MergeFeatureLabel()
  }

  def createMergeFeatureLabelFeatureTransformer(): MergeFeatureLabelFeatureTransformer = {
    new MergeFeatureLabelFeatureTransformer()
  }

  def createTFValidationMethod(valMethod: ValidationMethod[Float], name: String,
                               outputIndices: java.util.List[Int],
                               labelIndices: java.util.List[Int]): TFValidationMethod = {
    new TFValidationMethod(valMethod, name, outputIndices, labelIndices)
  }

  def createStatelessMetric(name: String, idx: Int): StatelessMetric = {
    new StatelessMetric(name, idx)
  }

  def createGanOptimMethod(dOptim: OptimMethod[T],
                           gOptim: OptimMethod[T],
                           dStep: Int, gStep: Int, gParamSize: Int): OptimMethod[T] = {
    new GanOptimMethod[T](dOptim, gOptim, dStep, gStep, gParamSize)
  }



  def createMiniBatchRDDFromStringRDD(stringRDD: JavaRDD[Array[Byte]],
                                     batchSize: Int): RDDWrapper[StringMiniBatch[T]] = {
    import TFTensorNumeric.NumericByteArray

    val rdd = stringRDD.rdd.mapPartitions { stringIter =>
      stringIter.grouped(batchSize).map { data =>
        val tensor = Tensor[Array[Byte]](data.toArray, shape = Array(data.length))
        new StringMiniBatch[T](tensor)
      }
    }
    RDDWrapper[StringMiniBatch[T]](rdd)
  }

  def createRDDFromTFRecords(path: String,
                             jsc: JavaSparkContext,
                             serializedParseGraph: Array[Byte],
                             inputName: String,
                             outputNames: JList[String]): RDD[Sample] = {
    val sc = jsc.sc

    val bserializedParseGraph = sc.broadcast(serializedParseGraph)
    val sampleRdd = sc.newAPIHadoopFile[org.apache.hadoop.io.BytesWritable,
      org.apache.hadoop.io.NullWritable,
      org.tensorflow.hadoop.io.TFRecordFileInputFormat](path).map { KV =>
      KV._1.copyBytes()
    }.mapPartitions { iter =>
      val graphDef = bserializedParseGraph.value
      val g = new Graph()
      g.importGraphDef(graphDef)
      val sess = new Session(g)

      def addFetches(names: JList[String], runner: Session#Runner) = {
        var j = 0
        while (j < names.size()) {
          runner.fetch(names.get(j))
          j += 1
        }
      }

      def getFetches(results: JList[TTensor[_]]) = {
        val tensors = new java.util.ArrayList[JTensor](results.size())
        var j = 0
        while (j < results.size()) {
          val t = results.get(j)
          tensors.add(tfTensor2JTensor(t))
          j += 1
        }
        tensors
      }


      val records = iter.toArray
      val samples = new Array[Sample](records.length)
      var i = 0

      while (i < records.length) {

        val bytes = records(i)
        val input = TTensor.create(bytes)
        val runner = sess.runner()
        runner.feed(inputName, input)
        addFetches(outputNames, runner)
        val results = runner.run()
        val outputTensors = getFetches(results)

        input.close()
        var j = 0
        while (j < results.size()) {
          results.get(j).close()
          j += 1
        }

        samples(i) = Sample(outputTensors, new java.util.ArrayList[JTensor], "float")

        i += 1
      }

      sess.close()
      g.close()

      samples.toIterator
    }

    sampleRdd
  }

  private def tfTensor2JTensor(t: TTensor[_]): JTensor = {
    val shape = t.shape().map(_.toInt)
    val length = shape.product
    val data = new Array[Float](length)
    val buffer = FloatBuffer.wrap(
      data,
      0,
      length)
    t.writeTo(buffer)
    JTensor(data, shape, "float")
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

  def torchNetSavePytorch(torchnet: TorchNet, path: String): Unit = {
    torchnet.savePytorch(path)
  }

}
