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

package com.intel.analytics.bigdl.optim

import java.util.concurrent.{LinkedBlockingQueue, ThreadPoolExecutor, TimeUnit}

import com.intel.analytics.bigdl.dataset.RDDDataSet
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.ps.ParameterManager
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Activities, T, Table}
import org.apache.spark.Logging

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect.ClassTag

object RDDOptimizer {
  private var lossArray: Array[Double] = null
  private var recordsArray: Array[Int] = null

  private val maxThread = System.getProperty(
    "com.intel.analytics.bigdl.optim.RDDOptimizer.maxThread",
    (Runtime.getRuntime().availableProcessors() * 50 / 2).toString()).toInt

  val context = new ExecutionContext {
    val threadPool =
      new ThreadPoolExecutor(maxThread, maxThread, 0L, TimeUnit.MILLISECONDS,
        new LinkedBlockingQueue[Runnable])

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable) {}
  }

  var thread: Thread = null
}

class RDDOptimizer[T: ClassTag](
  module: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optm: OptimMethod[T],
  pm: ParameterManager[T],
  dataSet: RDDDataSet[(Tensor[T], Tensor[T])],
  endWhen: Trigger,
  metrics: Metrics,
  subModuleNumber: Int,
  state: Table = T())
  (implicit ev: TensorNumeric[T])
  extends Optimizer[T](endWhen) with Logging {

  val sc = dataSet.data().sparkContext

  import RDDOptimizer._

  private def initThreadModules() = {
    val broadcast = sc.broadcast((module, criterion, state))
    val _subModuleNumber = subModuleNumber
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion, broadcastState) = broadcast.value
      val test = (0 until _subModuleNumber).map { _ =>
        val localModule = broadcastModule.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val (weights, grads) = localModule.getParameters()
        CachedModel(localModule, localCriterion, weights, grads, localState)
      }
      Iterator(test.toArray)
    }).persist()
    models.setName("Thread Model RDD")
    logInfo("Cache thread models...")
    models.count()
    logInfo("Cache thread models... done")
    models
  }

  val multiThreadModels = initThreadModules()

  private def initWorkerModules() = {
    val broadcast = sc.broadcast((module, criterion))
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      val localModule = broadcastModule.cloneModule()
      val localCriterion = broadcastCriterion.cloneCriterion()
      val (weights, grads) = localModule.getParameters()
      Iterator.single(CachedModel(localModule, localCriterion, weights, grads, T()))
    }).persist()
    models.setName("Worker Model RDD")
    logInfo("Cache worker models...")
    models.count()
    logInfo("Cache worker models... done")
    models
  }

  val models = initWorkerModules()


  override def optimize(): Module[Activities, Activities, T] = {
    // don't send whole Optimizer in closure
    val broadcastEV = sc.broadcast(ev)
    val partitionNum = dataSet.partitions().partitions.length
    var wallClockTime = 0L
    var epoch = state.get[Int]("epoch").getOrElse(1)
    var iter = state.get[Int]("neval").getOrElse(1)
    val _subModuleNumber = subModuleNumber
    var accumulateCount = 0
    val shufflebefore = System.nanoTime()
    logInfo(s"config $state")
    logInfo(s"Shuffle data")
    dataSet.shuffle()
    val shuffleEnd = System.nanoTime()
    logInfo(s"Shuffle data complete. Takes ${(shuffleEnd - shufflebefore) / 1e9}s")
    var epochStart = System.nanoTime()
    while (endWhen(state)) {
      val _header = header(epoch, accumulateCount, dataSet.size(), iter,
        wallClockTime)
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      val stackCount = sc.accumulator(0, "stack count")
      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("init gradient time", 0.0, sc, partitionNum)
      metrics.set("construct tensor time", 0.0, sc, partitionNum)
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("prepare time", 0.0, sc, partitionNum)
      metrics.set("statics time", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)

      val driverMetrics = metrics
      val start = System.nanoTime()
      val resultRDD = dataSet.data().zipPartitions(
        models,
        pm.sync(models.mapPartitions(iter => Iterator.single(iter.next().weight))),
        multiThreadModels, true)(
        (data, modelIter, weights, multiThreadModuleIter) => {
          var tmp = System.nanoTime()

          val localMTCaches = multiThreadModuleIter.next()
          val localCaches = modelIter.next()
          val syncWeightTask = Future {
            weights.next() // Update local weights
            (0 until _subModuleNumber).map(i => Future {
              localMTCaches(i).weight.copy(localCaches.weight)
            }(context)).foreach(Await.result(_, Duration.Inf))
          }(context)

          val localEV = broadcastEV.value
          tmp = System.nanoTime()
          if (thread != null) {
            thread.join()
            thread = null
          }
          driverMetrics.add("init gradient time", System.nanoTime() - tmp)

          tmp = System.nanoTime()
          val tensorBuffer = new Array[(Tensor[T], Tensor[T])](_subModuleNumber)
          val constructTensorTask = Future {
            val batch = data.next()
            var b = 0
            require(batch._1.size(1) == batch._2.size(1))
            require(batch._1.size(1) % _subModuleNumber == 0)
            val stackSize = batch._1.size(1) / _subModuleNumber
            while (b < _subModuleNumber) {
              tensorBuffer(b) = (batch._1.narrow(1, b * stackSize + 1, stackSize),
                batch._2.narrow(1, b * stackSize + 1, stackSize))
              b += 1
            }
          }(context)

          Await.result(constructTensorTask, Duration.Inf)
          driverMetrics.add("construct tensor time", System.nanoTime() - tmp)
          Await.result(syncWeightTask, Duration.Inf)

          driverMetrics.add("prepare time", System.nanoTime() - tmp)

          if (lossArray == null) {
            lossArray = new Array[Double](_subModuleNumber)
          }

          if (recordsArray == null) {
            recordsArray = new Array[Int](_subModuleNumber)
          }

          // ======================Start train models===================================
          tmp = System.nanoTime()
          (0 until _subModuleNumber).map(i => Future {
            val localModule = localMTCaches(i).model
            localModule.training()
            val localCriterion = localMTCaches(i).criterion
            val (inputFloat, targetFloat) = tensorBuffer(i)
            val input = inputFloat.asInstanceOf[Tensor[T]]
            val target = targetFloat.asInstanceOf[Tensor[T]]
            val output = localModule.forward(input)
            lossArray(i) = localEV.toType[Double](localCriterion.forward(output, target))
            val errors = localCriterion.backward(output, target)
            localModule.backward(input, errors)
            recordsArray(i) = target.size(1)
          }(context)).foreach(Await.result(_, Duration.Inf))
          val computingTime = System.nanoTime() - tmp
          driverMetrics.add("computing time average", computingTime)
          driverMetrics.add("computing time for each node", computingTime)
          tmp = System.nanoTime()
          stackCount += tensorBuffer.size
          var i = 0
          while (i < lossArray.length) {
            lossSum += lossArray(i)
            recordsNum += recordsArray(i)
            i += 1
          }
          driverMetrics.add("statics time", System.nanoTime() - tmp)

          tmp = System.nanoTime()
          val grads = localMTCaches.map(_.gradient)
          val gradLength = grads(0).nElement()
          val taskSize = gradLength / _subModuleNumber
          val extraTask = gradLength % _subModuleNumber

          val parallelNum = if (taskSize == 0) extraTask else _subModuleNumber
          (0 until parallelNum).map(tid => Future {
            val offset = tid * taskSize + math.min(tid, extraTask)
            val length = taskSize + (if (tid < extraTask) 1 else 0)
            var i = 0
            while (i < grads.length) {
              if (i == 0) {
                localCaches.gradient.narrow(1, offset + 1, length)
                  .copy(grads(i).narrow(1, offset + 1, length))
              } else {
                localCaches.gradient.narrow(1, offset + 1, length)
                  .add(grads(i).narrow(1, offset + 1, length))
              }
              i += 1
            }
          }(context)).foreach(Await.result(_, Duration.Inf))
          driverMetrics.add("aggregate gradient time", System.nanoTime() - tmp)

          thread = new Thread(new Runnable {
            override def run(): Unit = {
              (0 until _subModuleNumber).map(i => Future {
                localMTCaches(i).model.training()
                localMTCaches(i).model.zeroGradParameters()
              }(context)).foreach(Await.result(_, Duration.Inf))
            }
          })
          thread.start()

          Iterator.single(localCaches.gradient)
        })
      val reduceBefore = System.nanoTime()
      val driverEV = ev
      val optM = optm
      val driverParNum = partitionNum * _subModuleNumber
      pm.sumAndUpdate(resultRDD, (weights, gradients, state) => {
        gradients.div(driverEV.fromType[Int](driverParNum))
        state("neval") = iter
        state("epoch") = epoch
        optM.optimize(_ => (driverEV.fromType(lossSum.value / stackCount.value), gradients),
          weights, state, state)
      })
      val reduceAfter = System.nanoTime()

      accumulateCount += recordsNum.value
      val end = System.nanoTime()
      logInfo(s"${_header} Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
        s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
          lossSum.value / stackCount.value
        }. " +
        s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. ")
      logInfo("\n" + metrics.summary())
      iter += 1
      if (accumulateCount >= dataSet.size()) {
        val epochEnd = System.nanoTime()
        wallClockTime = wallClockTime + epochEnd - epochStart
        epochStart = System.nanoTime()
        logInfo(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")

        epoch += 1
        dataSet.reset()
        dataSet.shuffle()
        accumulateCount = 0
      }

      validate(wallClockTime, iter, epoch)
      cache(wallClockTime, iter, epoch)
    }
    validate(wallClockTime, iter, epoch)
    cache(wallClockTime, iter, epoch)

    module.asInstanceOf[Module[Activities, Activities, T]]
  }

  private def cache(wallClockTime: Long, iter: Int, epoch: Int): Unit = {
    val state = T("neval" -> iter, "epoch" -> epoch)
    cacheTrigger.foreach(trigger => {
      if (trigger(state) && cachePath.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
        saveModel(getModule().asInstanceOf[Module[Activities, Activities, T]],
          s".${state[Int]("neval")}")
        saveState(pm.getState(), s".${state[Int]("neval")}")
      }
    })
  }

  private def validate(wallClockTime: Long, iter: Int, epoch: Int): Unit = {
    val state = T("neval" -> iter, "epoch" -> epoch)
    validationTrigger.foreach(trigger => {
      if (trigger(state) && validator.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
        val results = validator.get.validate(getModule()
          .asInstanceOf[Module[Activities, Activities, T]])
        results.foreach(r => {
          logInfo(s"${r._1} is ${r._2}")
        })
      }
    })
  }

  private def getModule(): Module[Tensor[T], Tensor[T], T] = {
    multiThreadModels.first().head.model
  }
}


