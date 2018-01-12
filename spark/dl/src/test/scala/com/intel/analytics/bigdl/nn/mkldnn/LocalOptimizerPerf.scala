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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.Inception_v2
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, Table, ThreadPool}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object LocalOptimizerPerf {
  val modelSupported = Set("inception_v1","inception_v2", "vgg16", "vgg19", "alexnet", "resnet_50",
    "lstm", "lstmpeephole", "simplernn", "gru", "convlstmpeephole")
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
      .text(s"Model name. It can be ${modelSupported.mkString("| ")}")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if (modelSupported.contains(v.toLowerCase())) {
          success
        } else {
          failure(s"Model name only supports ${modelSupported.mkString(" | ")}")
        }
      )
  }

  def getModel(module: String,
               batchSize: Int): (Module[Float], MiniBatch[Float], Criterion[Float]) = {
    val (_model, input, criterion) = module match {
      case "alexnet" =>
        (AlexNet(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
      case "alexnetDnn" =>
        (DnnUtils.dnnAlexNet(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227).randn(),
          Tensor(batchSize).fill(1)), ClassNLLCriterion())
    }
    (_model, input, criterion)
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {


    def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
      Long, Long)]): Unit = {
      var forwardSum = 0L
      var backwardSum = 0L
      times.foreach(x => {
        forwardSum += x._2
        backwardSum += x._3
      })
      println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

      val all = forwardSum + backwardSum

      val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
        _ <: Activity, Float], Long, Long, Long, Double)]
      var i = 0
      while (i < times.length) {
        val all = times(i)._2 + times(i)._3
        val rate = times(i)._3.toDouble/ times(i)._2
        timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate))
        i += 1
      }
      val sortData = timeBuffer.sortBy(a => a._4)
      sortData.foreach(println)
    }

    def all(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      val timeBuffer =
        new ArrayBuffer[(AbstractModule[_ <: Activity, _ <: Activity, Float], Long, Long, Double)]

      for (i <- 0 to param.iteration) {
        val start = System.nanoTime()

        var b = 0
        val stackSize = input.size(1) / subModelNumber
        val extraSize = input.size(1) % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val inputBuffer = new Array[Tensor[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          inputBuffer(b) = input.narrow(1, offset, length)
          b += 1
        }

        val lossSum = default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
            () => {
              // println(s"running model ${i}")
              val localModel = workingModels(i)
              localModel.zeroGradParameters()
              localModel.training()
              val t1 = System.nanoTime()
              val output = localModel.forward(inputBuffer(i))
              val end1 = System.nanoTime() - t1
              val t2 = System.nanoTime()
              localModel.backward(inputBuffer(i), output)
              val end2 = System.nanoTime() - t2
              val tmp = localModel.getTimes()
              getTopTimes(tmp)
              localModel.resetTimes()
              // println("forward: " + end1 + " backward: " + end2 + " rate: " + end2.toDouble/end1)
              println("forward: " + end1 + " backward: " + end2 + " rate: " + end2.toDouble/end1)
            })
        )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
        )
      }
    }

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

    val optimizer = Optimizer(model, dummyDataSet, criterion)
    optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
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
    iteration: Int = 80,
    dataType: String = "float",
    module: String = "alexnetDnn" // "alexnetDnn"
  )
