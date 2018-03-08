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
package com.intel.analytics.bigdl.nn.mkldnn

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.all
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v1_NoAuxClassifier, Inception_v2}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Utils, Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Optimizer._
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.sql.execution.streaming
import org.apache.spark.sql.execution.streaming.state
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object LocalOptimizerPerf {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[LocalOptimizerPerfParam]("BigDL Local Performance Test") {
    head("Performance Test of Local Optimizer")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('c', "coreNumber")
      .text("physical cores number of current machine")
      .action((v, p) => p.copy(coreNumber = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[String]('m', "model")
      .text(s"Model name")
      .action((v, p) => p.copy(module = v))
    opt[Boolean]('t', "model")
      .text(s"Model name")
      .action((v, p) => p.copy(trainModel = v))
    opt[Boolean]("debug")
      .text(s"need Debug")
      .action((v, p) => p.copy(needDebug = v))
    opt[Int]('n', "numThreads")
      .text(s"numThreads")
      .action((v, p) => p.copy(numThreads = v))
  }

  def getModel(module: String,
               batchSize: Int): (Module[Float], MiniBatch[Float], Criterion[Float]) = {
    val (_model, input, criterion) = module match {
      case "alexnet" =>
        (AlexNet(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "alexnetDnn" =>
        (AlexNet.dnn(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "dnnModel" =>
        (DnnTools.dnnModel(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227).randn(),
          Tensor(batchSize, 96, 27, 27).randn()), ClassNLLCriterion())
      case "inception_v1" =>
        (Inception_v1(1000, false), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "inception_v1_dnn" =>
        (Inception_v1_dnn(1000, true), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "inception_no_dnn" =>
        (Inception_v1_NoAuxClassifier_dnn(1000, true),
          MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
            Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "inception_no" =>
        (Inception_v1_NoAuxClassifier(1000, true),
          MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
            Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "inception_v2" =>
        (Inception_v2(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "inception_v2_dnn" =>
        (Inception_v2_dnn(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "vgg16" =>
        (Vgg_16(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "vgg16_dnn" =>
        (Vgg_16_dnn(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "vgg19" =>
        println("vgg19")
        (Vgg_19(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "vgg19_dnn" =>
        println("vgg19 mkldnn")
        (Vgg_19_dnn(1000), MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "resnet_50" =>
        val model = ResNet(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> DatasetType.ImageNet))
        ResNet.shareGradInput(model)
        ResNet.modelInit(model)
        (model, MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), CrossEntropyCriterion())

      case "resnet_50_dnn" =>
        val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> ResNet_dnn.DatasetType.ImageNet))
//        ResNet_dnn.shareGradInput(model)
//        ResNet_dnn.modelInit(model)
        (model, MiniBatch(Tensor(batchSize, 3, 224, 224).randn(),
          Tensor(batchSize).fill(1)), CrossEntropyCriterion())
    }
    _model.createDnnEngine(0)
    _model.createStream()
    (_model, input, criterion)
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {

    def all(model: Module[Float], dataset: LocalDataSet[MiniBatch[Float]], iteration: Int): Unit = {

      val coreNumber = Engine.coreNumber()
      val subModelNumber = Engine.getEngineType match {
        case MklBlas => coreNumber
        case _ => throw new IllegalArgumentException
      }

      val workingModels = {
        model.getParameters()
        val wb = Util.getAndClearWeightBias(model.parameters())

        val models = (1 to subModelNumber).map(i => {
          logger.info(s"Clone $i model...")
          val m = model.cloneModule()
          Util.putWeightBias(wb, m)
          Util.initGradWeightBias(wb, m)
          // for dnn create engine and stream
          m.createDnnEngine(0)
          m.createStream()
          m
        }).toArray
        Util.putWeightBias(wb, model)
        Util.initGradWeightBias(wb, model)
        models
      }

      var wallClockTime = 0L
      var count = 0
      var iter = dataset.data(train = false)
      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      var i = 0
      while (i < iteration) {
        val start = System.nanoTime()
        // Fetch data and prepare tensors
        val batch = iter.next()
        var b = 0
        val stackSize = batch.size() / subModelNumber
        val extraSize = batch.size() % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val miniBatchBuffer = new Array[MiniBatch[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          miniBatchBuffer(b) = batch.slice(offset, length)
          b += 1
        }
        val dataFetchTime = System.nanoTime()
        val lossSum = Engine.default.invokeAndWait(
          (0 until parallelism).map(i =>
            () => {
              val s1 = System.nanoTime()
              val localModel = workingModels(i)
              val input = miniBatchBuffer(i).getInput()
              val output = localModel.forward(input)
              output.asInstanceOf[Tensor[Float]].setPrimitiveDesc(0L)
              localModel.backward(input, output)
              val e1 = (System.nanoTime() - s1)/1e6
              // println(s"thread time ${e1}")
              1
            })
        ).sum

        val end = System.nanoTime()
        logger.info(
          s"data fetch time is ${(dataFetchTime - start) / 1e9}s, " +
            s"train time ${(end - dataFetchTime) / 1e9}s. " +
            s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second. "
        )

        i += 1
      }
    }

    def time(model: Module[Float], dataset: MiniBatch[Float], iteration: Int): Unit = {
      val input = dataset.getInput()
      val target = dataset.getTarget()

      var warmup = 10
      for (i <- 1 to warmup) {
        val output = model.forward(input)
         model.backward(input, output)
      }

      println("start time")
      val s1 = System.nanoTime()
      for (i <- 1 to iteration) {
        val output = model.forward(input)
         model.backward(input, output)
//        val tmp = model.getTimes()
//        DnnTools.getTopTimes(tmp)
         println("11111111111111")
//        model.resetTimes()
      }
      val e1 = (System.nanoTime() - s1)/iteration
      println(s"average time cost ${e1/1e9}")
    }

    System.setProperty("bigdl.mklNumThreads", param.numThreads.toString)
    Engine.setCoreNumber(param.coreNumber)

    val (_model, miniBatch, criterion) = getModel(param.module, param.batchSize)
    val model = _model
    println(param.coreNumber)

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          private val index = new AtomicInteger()
          override def hasNext: Boolean = {
            if (train) {
              true
            } else {
              index.get() < 100000
            }
          }
          override def next(): MiniBatch[Float] = {
            index.getAndIncrement()
            miniBatch
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    if (param.trainModel) {
      val optimizer = Optimizer(model, dummyDataSet, criterion)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
    } else {
      if (param.needDebug) {
        System.setProperty("debug", "2")
        time(model, miniBatch, param.iteration)
      } else {
        all(model, dummyDataSet, param.iteration)
        // time(model, miniBatch, param.iteration)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new LocalOptimizerPerfParam()).foreach(performance)
  }
}

/**
* Local Optimizer Performance Parameters
*
* @param batchSize batch size
* @param coreNumber core number
* @param iteration how many iterations to run
* @param dataType data type (double / float)
* @param module module name
  */
case class LocalOptimizerPerfParam(
    batchSize: Int = 16,
    coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
    iteration: Int = 50,
    dataType: String = "float",
    module: String = "resnet_50_dnn", // "alexnetDnn"
    trainModel: Boolean = false,
    numThreads: Int = 1,
    needDebug: Boolean = false
  )
