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

import java.util
import java.util.concurrent.{Callable, ExecutorService, Executors, Future}
import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.all
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.Inception_v2
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
import spire.syntax.module

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object LocalOptimizerPerf2 {
  val modelSupported = Set("inception_v1","inception_v2", "vgg16", "vgg19", "alexnet", "resnet_50",
    "lstm", "lstmpeephole", "simplernn", "gru", "convlstmpeephole")
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[LocalOptimizerPerfParam2]("BigDL Local Performance Test") {
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

  def performance(param: LocalOptimizerPerfParam2): Unit = {

    def all(model: Module[Float], dataset: LocalDataSet[MiniBatch[Float]], iteration: Int): Unit = {
      val coreNumber = Engine.coreNumber()
      val subModelNumber = Engine.getEngineType match {
        case MklBlas => coreNumber
        case _ => throw new IllegalArgumentException
      }

      val workingModels = {
        val models = (1 to subModelNumber).map(i => {
          logger.info(s"Clone $i model...")
          val m = AlexNet(1000)
          // for dnn create engine and stream
          m.createDnnEngine(0)
          m.createStream()
          m
        }).toArray
        models
      }

      val default: ThreadPool = new ThreadPool(param.coreNumber)
      var wallClockTime = 0L
      var count = 0
      var iter = dataset.data(train = false)
      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      // Fetch data and prepare tensors
      val batch = iter.next()
      var b = 0
      val parallelism = subModelNumber
      val miniBatchBuffer = new Array[MiniBatch[Float]](parallelism)
      while (b < parallelism) {
        miniBatchBuffer(b) = MiniBatch(Tensor(4, 3, 227, 227).randn(), Tensor(4, 1000).randn())
        b += 1
      }
      var i = 0
      while (i < iteration) {
        val start = System.nanoTime()
        val dataFetchTime = System.nanoTime()
        val lossSum = default.invokeAndWait(
          (0 until parallelism).map(i =>
            () => {
              val localModel = workingModels(i)
              val input = miniBatchBuffer(i).getInput()
              val output = localModel.forward(input)
              localModel.backward(input, output)
              1
            })
        ).sum

        // println("lossSum " + lossSum)
        val end = System.nanoTime()
        logger.info(
          s"data fetch time is ${(dataFetchTime - start) / 1e9}s, " +
            s"train time ${(end - dataFetchTime) / 1e9}s. " +
            s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second. "
        )

        i += 1
      }
    }

    class TaskWithResult extends Callable[String] {
      val model = DnnUtils.dnnAlexNet(1000)
      model.createDnnEngine(0)
      model.createStream()
      val batchSize = 4
      val input = Tensor(batchSize, 3, 227, 227).randn()
      val gradOutput = Tensor(batchSize, 1000).randn()
      /**
        * 任务的具体过程，一旦任务传给ExecutorService的submit方法，
        * 则该方法自动在一个线程上执行
        */
      @throws[Exception]
      def call: String = {
        // 该返回结果将被Future的get方法得到
        val output = model.forward(input)
        model.backward(input, output)
        return "Thread id 是: " + Thread.currentThread().getName()
      }
    }

    def allNew(model: Module[Float], dataset: LocalDataSet[MiniBatch[Float]],
               iteration: Int): Unit = {
      import scala.collection.JavaConversions._
      val threadSize: Int = 4
      val executorService = Executors.newFixedThreadPool(threadSize)
      val taskList = new util.ArrayList[TaskWithResult]
      for (i <- 1 to threadSize) {
         taskList.add(new TaskWithResult)
      }
      // warm up
      for (j <- 0 to 6) {
          val resultList: util.List[Future[String]] = new util.ArrayList[Future[String]]
          for (i <- 0 to threadSize - 1) {
              val future = executorService.submit(taskList.get(i))
              resultList.add(future)
          }
          for (fs <- resultList) {
            while (!fs.isDone) {}
          }
      }

      var j = 0
      val start = System.nanoTime()
      while (j < iteration) {
        val s1: Long = System.nanoTime()
        val resultList: util.List[Future[String]] = new util.ArrayList[Future[String]]
        for (i <- 0 to threadSize - 1) {
            val future = executorService.submit(taskList.get(i))
            resultList.add(future)
        }
        for (fs <- resultList) {
          while (!fs.isDone) {}
          // println(fs.get())
        }
        val e1: Long = System.nanoTime() - s1
        // System.out.println("iteration time cost " + e1/1e9 + " s")
        j += 1
      }
      val end = System.nanoTime() - start
      System.out.println("total time cost " + end/1e9 + " s")
      executorService.shutdown()
    }

    // System.setProperty("bigdl.mklNumThreads", "4")
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

    if (false) {
      val optimizer = Optimizer(model, dummyDataSet, criterion)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
    } else {
      all(model, dummyDataSet, param.iteration)
      // allNew(model, dummyDataSet, param.iteration)
    }
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new LocalOptimizerPerfParam2()).foreach(performance)
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
case class LocalOptimizerPerfParam2(
    batchSize: Int = 16, // 16,
    coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
    iteration: Int = 80,
    dataType: String = "float",
    module: String = "alexnetDnn" // "alexnetDnn"
  )
