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

package com.intel.analytics.sparkdl.optim

import java.util.concurrent.{TimeUnit, ThreadPoolExecutor, LinkedBlockingQueue}

import com.intel.analytics.sparkdl.nn.{Criterion, Module}
import com.intel.analytics.sparkdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.sparkdl.ps.ParameterManager
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{T, Table}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}
import scala.reflect.ClassTag

object BetterGradAggEpochOptimizer {
  val subModuleNumber = System.getProperty(
    "com.intel.analytics.sparkdl.optim.BetterGradAggEpochOptimizer.subModuleNumber",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  val lossArray = new Array[Double](subModuleNumber)
  val recordsArray = new Array[Int](subModuleNumber)

  private val maxThread = System.getProperty(
    "com.intel.analytics.sparkdl.optim.BetterGradAggEpochOptimizer.maxThread",
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

class BetterGradAggEpochOptimizer[T: ClassTag](
  @transient module: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optm: OptimMethod[T],
  pm: ParameterManager[T],
  dataSets: DataSet[_, T] with HasEpoch,
  metrics: Metrics,
  config: Table = T())
  (implicit ev: TensorNumeric[T])
  extends EpochOptimizer[T](module, criterion, optm, pm, dataSets, metrics, config) {

  import BetterGradAggEpochOptimizer._

  private def init() = {
    val broadcast = dataSet.getSparkContext().broadcast((module, criterion))
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      val test = (0 until subModuleNumber).map { _ =>
        val localModule = broadcastModule.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val (weights, grads) = localModule.getParameters()
        CachedModel(localModule, localCriterion, weights, grads, T())
      }
      Iterator(test.toArray)
    }).persist()
    models.setName("modelRDD")
    logInfo("Cache models...")
    models.count()
    logInfo("Cache models... done")
    models
  }

  val multiThreadModels = init()


  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    // don't send whole Optimizer in closure
    val broadcastEV = dataSets.getSparkContext().broadcast(ev)

    val sc = dataSets.getSparkContext()
    val partitionNum = dataSets.getPartitionNum()
    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(20)
    val state = T()
    for (i <- 1 to epochNum) {
      logInfo(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()

      logInfo("config" + config)

      logInfo(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logInfo(s"[Epoch $i/$epochNum] Shuffle data complete. Takes ${
        (shuffleEnd -
          epochStart) / 1e9
      }s")
      config("epoch") = i
      while (!dataSets.epochFinished()) {
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
        val resultRDD = dataSets.fetch().zipPartitions(
          models,
          pm.sync(models.mapPartitions(iter => Iterator.single(iter.next().weight))),
          multiThreadModels, true)(
          (data, modelIter, weights, multiThreadModuleIter) => {
            var tmp = System.nanoTime()

            val localMTCaches = multiThreadModuleIter.next()
            val localCaches = modelIter.next()
            val syncWeightTask = Future {
              weights.next() // Update local weights
              (0 until subModuleNumber).map(i => Future {
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
            val tensorBuffer = new Array[(Tensor[T], Tensor[T])](subModuleNumber)
            val constructTensorTask = Future {
              val batch = data.next()
              var b = 0
              while (b < subModuleNumber) {
                tensorBuffer(b) = batch.next()
                b += 1
              }
            }(context)

            Await.result(constructTensorTask, Duration.Inf)
            driverMetrics.add("construct tensor time", System.nanoTime() - tmp)
            Await.result(syncWeightTask, Duration.Inf)

            driverMetrics.add("prepare time", System.nanoTime() - tmp)

            // ======================Start train models===================================
            tmp = System.nanoTime()
            (0 until subModuleNumber).map(i => Future {
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
            val taskSize = gradLength / subModuleNumber
            val extraTask = gradLength % subModuleNumber

            (0 until subModuleNumber).map(tid => Future {
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
                (0 until subModuleNumber).map(i => Future {
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
        val configDriver = config
        val driverParNum = partitionNum * subModuleNumber
        pm.sumAndUpdate(resultRDD, (weights, gradients, state) => {
          gradients.div(driverEV.fromType[Int](driverParNum))
          optM.optimize(_ => (driverEV.fromType(lossSum.value / stackCount.value), gradients),
            weights, configDriver, state)
        })
        val reduceAfter = System.nanoTime()

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        logInfo(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${
          recordsNum.value
        } in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
            lossSum.value / stackCount.value
          }. " +
          s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. ")
        logInfo("\n" + metrics.summary())
      }
      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logInfo(s"[Epoch $i/$epochNum] Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      saveState(pm.getState(), i)
      test(module, i)
    }

    saveModel(module)
    saveState(pm.getState())
    module
  }

  override val evaluateModels =
    multiThreadModels.mapPartitions(iter => {
      require(iter.hasNext)
      Iterator.single(iter.next()(0))
    })
}

